import { AbstractMatrix, Matrix } from 'ml-matrix';
import {
  LassoRegressionOptions,
  Stats,
  defaultOptions,
  LassoRegressionConfig,
} from './types';
import { standardize, unStandardizeWeights, getMaxVectorLength } from './utils';

export class LassoRegression {
  importance?: number[][];
  weights?: number[][];
  options?: LassoRegressionOptions;
  xStats?: Stats;
  yStats?: Stats;
  converged?: boolean;

  /**
   * @param xInput 2D array of x
   * @param yInput 2D array of y
   * @param options.lambda lambda hyper-parameter
   * @param options.tolerance max tolerance for change in weights
   * @param options.maxIter max iterations (to avoid infinite loop)
   */

  constructor(
    xInput: Array<number[]> | boolean,
    yInput: Array<number[]> | LassoRegressionConfig,
    options: Partial<LassoRegressionOptions>
  ) {
    //reloading a model from JSON if xInput is true and yInput is an object.
    if (
      xInput === true &&
      Object.prototype.toString.call(yInput) !== '[object Array]'
    ) {
      const config = yInput as LassoRegressionConfig;
      this.importance = config.importance;
      this.weights = config.weights;
      this.options = config.options;
      this.xStats = config.xStats;
      this.yStats = config.yStats;
      this.converged = config.converged;
    }
    //if xInput and yInputs are both arrays
    if (
      Object.prototype.toString.call(xInput) === '[object Array]' &&
      Object.prototype.toString.call(yInput) === '[object Array]'
    ) {
      //create matrices from 2D array inputs
      const x = new Matrix(xInput as number[][]);
      const y = new Matrix(yInput as number[][]);
      //set default options if not defined
      this.options = { ...defaultOptions, ...options };
      //standarize the x and y matrices and get input stats.
      this.xStats = standardize(x, false);
      this.yStats = standardize(y, false);
      //run coordinate descent algorithm
      const { w, converged } = this._cd(
        x,
        y,
        this.options.maxIter,
        this.options.tolerance
      );
      this.importance = w.to2DArray();
      this.converged = converged;
      //get un-standardized coefficients and y-intercept.
      this.weights = unStandardizeWeights(
        w,
        this.xStats,
        this.yStats
      ).to2DArray();
    }
  }

  predict(xInput: number[][]) {
    if (this.weights === undefined) {
      throw new Error('model has not been calculated');
    }
    const x = new Matrix(xInput);
    const ones = Matrix.ones(x.rows, 1);
    x.addColumn(x.columns, ones);
    const w = new Matrix(this.weights);
    const y = x.mmul(w);
    return y.to2DArray();
  }

  _softThresholding(rho: AbstractMatrix, n: number, lambda: number) {
    for (let c = 0; c < rho.columns; c++) {
      const rhoi = rho.get(0, c);
      if (rhoi < -lambda * n) {
        rho.set(0, c, rhoi + n * lambda);
      } else if (rhoi > lambda * n) {
        rho.set(0, c, rhoi - lambda * n);
      } else {
        rho.set(0, c, 0);
      }
    }
  }

  _cd(
    x: AbstractMatrix,
    y: AbstractMatrix,
    maxIter: number,
    tolerance: number
  ) {
    //initialize the coefficients to zero
    const w = Matrix.zeros(x.columns, y.columns);
    let count = 0;
    let tol = 1;
    //get z = sum(xi^2); if sum(xi^2) is 0, then set to 1 as to not divide by 0;
    const z = Array(x.columns)
      .fill(1)
      .map((_, i) => {
        const xi = x.getColumnVector(i);
        const zi = xi.dot(xi);
        return zi === 0 ? 1 : zi;
      });
    while (count < maxIter && tol > tolerance) {
      const wPrev = w.clone();
      for (let i = 0; i < w.rows; i++) {
        let xi = x.getColumnVector(i);
        let wi = w.clone();
        for (let j = 0; j < wi.columns; j++) {
          wi.set(i, j, 0);
        }
        wi = x.mmul(wi);
        wi = Matrix.sub(y, wi);
        let rho = xi.transpose().mmul(wi);
        this._softThresholding(rho, x.rows, this.options?.lambda ?? 0);
        rho.div(z[i]);
        w.setRow(i, rho);
      }
      const wCurrent = w.clone();
      tol = getMaxVectorLength(wCurrent, wPrev);
      count++;
    }
    //if tolerance is greater than max tolerance, then the minimization did not converge.
    if (tol > tolerance) {
      return { w, converged: false };
    } else {
      return { w, converged: true };
    }
  }

  toJSON() {
    return {
      name: 'lassoRegression',
      weights: this.weights,
      importance: this.importance,
      options: this.options,
      xStats: this.xStats,
      yStats: this.yStats,
      converged: this.converged,
    };
  }

  static load(model: LassoRegressionConfig) {
    if (model.name !== 'lassoRegression') {
      throw new Error('not a LASSO model');
    }
    return new LassoRegression(true, model, {});
  }
}
