import { LassoRegression } from '../src';
import { x, y } from './data';

describe('lambda=0', () => {
  it('works', () => {
    const lasso = new LassoRegression(x, y, { lambda: 0 });
    //standardized solution.
    const LeastSquaresSolution = [
      [-0.62404352, -0.62404352],
      [-0.28258203, -0.28258203],
      [0, 0],
    ];
    expect(lasso.importance?.length).toBe(LeastSquaresSolution.length);
    lasso.importance?.forEach((row, r) => {
      expect(row.length).toBe(LeastSquaresSolution[r].length);
      row.forEach((wij, c) => {
        expect(wij).toBeCloseTo(LeastSquaresSolution[r][c], 4);
      });
    });
  });
});

describe('lambda=0.5', () => {
  it('works', () => {
    const lasso = new LassoRegression(x, y, { lambda: 0.5 });
    //standardized solution.
    const skLearnSolution = [
      [-0.34490134, -0.34490134],
      [-0.00336915, -0.00336915],
      [0, 0],
    ];
    expect(lasso.importance?.length).toBe(skLearnSolution.length);
    lasso.importance?.forEach((row, r) => {
      expect(row.length).toBe(skLearnSolution[r].length);
      row.forEach((wij, c) => {
        expect(wij).toBeCloseTo(skLearnSolution[r][c], 4);
      });
    });
  });
});

describe('lambda=0.8', () => {
  it('works', () => {
    const lasso = new LassoRegression(x, y, { lambda: 0.8 });
    //standardized solution.
    const skLearnSolution = [
      [-0.04755138, -0.04755138],
      [-0, -0],
      [0, 0],
    ];
    expect(lasso.importance?.length).toBe(skLearnSolution.length);
    lasso.importance?.forEach((row, r) => {
      expect(row.length).toBe(skLearnSolution[r].length);
      row.forEach((wij, c) => {
        expect(wij).toBeCloseTo(skLearnSolution[r][c], 4);
      });
    });
  });
});
