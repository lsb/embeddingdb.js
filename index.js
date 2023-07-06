// This is an embedding database for the browser.
// We can do tiled faceted embedding nearest neighbor search returning results in real time.
//
// This version 0.1 takes 384-dimension embeddings (like from a sentence-transformer),
// uses product quantization to map that into 48 7-bit codewords,
// writes out a codebook, arrow files for the embeddings, arrow files for the metadata,
// computes distances from an arbitrary 384-dimension point using ONNX,
// and performs a faceted top-k search using ONNX.
//
// An iPhone 14 can perform a distance computation for 0.1M distances in 10ms,
// and can find the top 10 nearest neighbors for 2M distances in 5ms while augmenting those distances with an arbitrary weight for the particular facet.
//
// When sharding the embeddings, shard sizes must be multiples of chunkSize, which is 100,000 by default.
//
// The distance computation calls a callback around every `maxTick` milliseconds to allow incremental feedback.
// Choosing a shard order appropriate to the business use case (like size of the page) will reduce jitter in the UI.


import { tableFromIPC } from 'apache-arrow';
import { pipeline } from '@xenova/transformers';
import * as ort from 'onnxruntime-web';
import {filteredTopKAsc, pqDist} from 'pq.js';

const env = {
    maxTick: 30,
    chunkSize: 100000,
}

const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

const distTopK = async function (inferenceSession, dists, filterColumn, filterValue, filterZero, filterShim, k) {
    const {output: {data: topk}} = await inferenceSession.run({
        "input": (new Tensor("float32", dists)),
        "filterColumn": (new Tensor("float32", filterColumn)),
        "filterValue": (new Tensor("float32", [filterValue])),
        "filterZero": (new Tensor("float32", [filterZero])),
        "filterShim": (new Tensor("float32", [filterShim])),
        "k": (new Tensor("uint8", [k])),
    });
    return topk;
}


const queryDist = async function (inferenceSession, query, codebook, codebookShape, embeddings, embeddingTensorShape) {
    const {output: {data: distTile}} = await inferenceSession.run({
        "query": (new Tensor("float32", query)),
        "codebook": (new Tensor("float32", codebook, codebookShape)),
        "embeddings": (new Tensor("uint8", embeddings, embeddingTensorShape)),  
    })
    return distTile;
}


const queryToTiledDist = async function (query, embeddings, codebk, codebkflat, pqdistinf, dists, firstLetters, firstLetterInt, filteredtopkinf, k, intermediateValueFn, continueFn, embeddingCounter=0) {
    // compute distances a chunk of an embedding shard at a time,
    // mutate the dists array,
    // compute topk not more frequently than every `maxTick` milliseconds to avoid jitter,
    // call a function to check if (external state has changed and) we should abandon this distance computation,
    // stream updates to the intermediateValue callback.

    let lastPaint = Date.now();
    const timingStrings = [];
    const codebookshape = [codebk.length, codebk[0].length, codebk[0][0].length];

    for(; embeddingCounter<embeddings.length; embeddingCounter++){
      const {data: embeddingData, offset: embeddingOffset} = embeddings[embeddingCounter];
      for(let i=0; i < (embeddingData.length / codebk.length) && continueFn(); i += env.chunkSize) {
        const startTime = Date.now();
        const startEmbeddingPosition = i * codebk.length;
        const embeddingTileLength = env.chunkSize * codebk.length;
        const embeddingTensorShape = [env.chunkSize, codebk.length];
        const embeddingTile = new Uint8Array(embeddingData.buffer, startEmbeddingPosition + embeddingData.byteOffset, embeddingTileLength);
        const distTile = await queryDist(pqdistinf, query, codebkflat, codebookshape, embeddingTile, embeddingTensorShape);
        for(let j = 0; j < env.chunkSize; j++) {
          dists[embeddingOffset+i+j] = distTile[j];
        }
        const distTime = Date.now();
        timingStrings.push(`${distTime - startTime}`)
        intermediateValueFn({dists, distTime: timingStrings.join(), lastPaint});
        if((i === 0 && embeddingCounter === 0) || (distTime - lastPaint > env.maxTick)) {
          const topk = await distTopK(filteredtopkinf, dists, firstLetters, firstLetterInt, 0, 1024, k);
          const topktime = Date.now();
          timingStrings.push(`-${topktime-distTime} `);
          intermediateValueFn({topk});
          await (new Promise(r => setTimeout(r,0))); // ensure that everything else waiting gets to run
          lastPaint = Date.now()
        }
      }
    }
    if(continueFn()) {
      const topk = await distTopK(filteredtopkinf, dists, firstLetters, firstLetterInt, 0, 1024, k);
      intermediateValueFn({dists, distTime: timingStrings.join(), topk});
    }
}


const makeONNXRunnables = async function () {
    const filteredtopkinf = await InferenceSession.create(filteredTopKAsc);
    const pqdistinf = await InferenceSession.create(pqDist, {executionProviders: ['wasm']});
    return {pqdistinf, filteredtopkinf};
}

const flattenCodebook = (codebk) => Float32Array.from(
    {length: codebk.length * codebk[0].length * codebk[0][0].length},
    (e,i) => codebk[Math.floor(i / codebk[0].length / codebk[0][0].length)][Math.floor(i / codebk[0][0].length) % codebk[0].length][i % codebk[0][0].length]
)

module.exports = {
    distTopK,
    queryDist,
    queryToTiledDist,
    makeONNXRunnables,
    flattenCodebook,
    env,
}