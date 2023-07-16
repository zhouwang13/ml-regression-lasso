export interface LassoRegressionOptions {
  lambda: number;
  tolerance: number;
  maxIter: number;
}

export interface Stats {
  mean: number[];
  std: number[];
}

export const defaultOptions: LassoRegressionOptions = {
  lambda: 0,
  tolerance: 0.00001,
  maxIter: 200,
};

export interface LassoRegressionConfig {
  name: 'lassoRegression';
  importance: number[][];
  weights: number[][];
  options: LassoRegressionOptions;
  xStats: Stats;
  yStats: Stats;
  converged: boolean;
}
