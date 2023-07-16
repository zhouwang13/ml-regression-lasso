# ml-regression-lasso

Lasso (least absolute shrinkage and selection operator) Regression

## Installation

`yarn add ml-regression-lasso`

## API

### new LassoRegression(x, y[, options])

**Arguments**

- `x`: Matrix containing the inputs
- `y`: Matrix containing the outputs

**Options**

- `lambda`: Constant that multiplies the L1 term, controlling regularization strength. Lambda must be a non-negative float i.e. in `[0,  inf)`(default:0)
- `tolerance`: The tolerance for the optimization: if the updates are smaller than `tol`, the coordinate descent algorithm is considered converged. (default: 0.00001)
- `maxIter`: The maximum number of iterations.(default: 200)

## Usage

```js
import LassoRegression from "ml-regression-lasso";
const  x = [
	[1,32,120],
	[2,34,150],
	[3,94,40],
	[4,54,2],
];
const  y = [
	[757,378.5],
	[944,472],
	[349,174.5],
	[86,43],
];
const  lasso = new  LassoRegression(x, y, { lambda:  0.1 });
console.log(lasso.weights);
/*
[
  [ 0, 0 ],
  [ 0, 0 ],
  [ 5.078686699741143, 2.5393433498705713 ],
  [ 137.86243742019087, 68.93121871009544 ]
]
*/
console.log(lasso.predict(x));
/*
[
  [ 747.304841389128, 373.652420694564 ],
  [ 899.6654423813623, 449.8327211906811 ],
  [ 341.00990540983656, 170.50495270491828 ],
  [ 148.01981081967315, 74.00990540983658 ]
]
*/
```

## Acknowledgments

This work was inspired by [mljs](https://github.com/mljs/regression-multivariate-linear).  Lasso regression was not in the included libraries, and I wasn't able to find one readily available that matched the results of scikitLearn.

The algorithm is based on this [article](https://www.kaggle.com/code/ddatad/coordinate-descent-for-lasso-normal-regression).

## License

[MIT](./LICENSE)