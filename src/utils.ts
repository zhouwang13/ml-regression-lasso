import { AbstractMatrix } from 'ml-matrix';
import { Stats } from './types';

export const standardize = (matrix: AbstractMatrix, unbiased: boolean) => {
  const mean = matrix.mean('column');
  const std = matrix.standardDeviation('column', { unbiased });
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.columns; j++) {
      // (xi-xmean)/std(x)
      // if x is a constant, then std = 0.  In this case, divide by 1 instead of 0;
      matrix.set(
        i,
        j,
        (matrix.get(i, j) - mean[j]) / (std[j] === 0 ? 1 : std[j])
      );
    }
  }
  return {
    mean,
    std,
  };
};

export const unStandardizeWeights = (w: AbstractMatrix, x: Stats, y: Stats) => {
  const beta = w.clone();
  for (let j = 0; j < beta.columns; j++) {
    for (let i = 0; i < beta.rows; i++) {
      beta.set(
        i,
        j,
        (beta.get(i, j) / (x.std[i] === 0 ? 1 : x.std[i])) * y.std[j]
      );
    }
  }
  const yi = y.std.map((sigmaY, j) => {
    const sW = w.clone().getColumn(j);
    const sumXFactors = sW.reduce(
      (acc, b, r) =>
        x.mean[r] !== undefined
          ? acc - (b * sigmaY * x.mean[r]) / (x.std[r] === 0 ? 1 : x.std[r])
          : acc,
      0
    );
    return y.mean[j] + sumXFactors;
  });
  beta.addRow(w.rows, yi);
  return beta;
};

export const getMaxVectorLength = (v1: AbstractMatrix, v2: AbstractMatrix) => {
  const dw = v2.sub(v1);
  let max = 0;
  for (let c = 0; c < v1.columns; c++) {
    const dwi = dw.getColumnVector(c);
    const length = Math.sqrt(dwi.dot(dwi));
    if (length > max) {
      max = length;
    }
  }
  return max;
};
