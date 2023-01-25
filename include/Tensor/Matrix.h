#pragma once

#include "Tensor/Vector.h"
#include "Tensor/Quat.h"

/*
Here's some common OpenGL / 3D matrix operations
*/

namespace Tensor {

//glTranslate
// TODO optimize for storage ... dense upper-right corner, rest is identity
template<typename real>
mat<real,4,4> translate(
	vec<real,3> t
) {
	return mat<real,4,4>{
		{1, 0, 0, t.x},
		{0, 1, 0, t.y},
		{0, 0, 1, t.z},
		{0, 0, 0, 1}
	};
}

//glScale
// TODO optimized storage ... diagonal scale-only ...
template<typename real>
mat<real,4,4> scale(
	vec<real,3> s
) {
	return mat<real,4,4>{
		{s.x, 0, 0, 0},
		{0, s.y, 0, 0},
		{0, 0, s.z, 0},
		{0, 0, 0, 1}
	};
}

//glRotate
template<typename real>
mat<real,4,4> rotate(
	real rad,
	vec<real,3> axis
) {
	auto q = quat<real>(axis.x, axis.y, axis.z, rad)
		.fromAngleAxis();
	auto x = q.xAxis();
	auto y = q.yAxis();
	auto z = q.zAxis();
/*
which is faster?
 this 4x4 mat mul?
 or quat-rotate the col vectors of mq?

TODO how about a sub-matrix specilized storage? dense upper-left, rest is diagonal.
where you can set a cutoff dimension -- and every index beyond that dimension is filled with identity.
So that this could just be stored as a 3x3's worth (9 reals) in memory, even though it's a 4x4
Likewise for asymmetric storage, 3x4 translations could be just 12 reals in memory.
*/
	return mat<real,4,4> {
		{x.x, y.x, z.x, 0},
		{x.y, y.y, z.y, 0},
		{x.z, y.z, z.z, 0},
		{0, 0, 0, 1}
	};
}

//gluLookAt
//https://stackoverflow.com/questions/21830340/understanding-glmlookat
template<typename real>
mat<real,4,4> lookAt(
	vec<real,3> eye,
	vec<real,3> center,
	vec<real,3> up
) {
	auto Z = (eye - center).normalize();
	auto Y = up;
	auto X = Y.cross(Z).normalize();
	Y = Z.cross(X);
	// could explot a submatrix-storage, rest-is-identity optimization ...
	return mat<real,4,4>{
		{X.x, X.y, X.z, -eye.dot(X)},
		{Y.x, Y.y, Y.z, -eye.dot(Y)},
		{Z.x, Z.y, Z.x, -eye.dot(Z)},
		{0, 0, 0, 1},
	};
}

/*
glFrustum
https://www.khronos.org/opengl/wiki/GluPerspective_code
https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html
*/
template<typename real>
mat<real,4,4> frustum(
	real left,
	real right,
	real bottom,
	real top,
	real near,
	real far
) {
	auto near2 = 2 * near;
	auto diff = vec<real,3>(right, top, far) - vec<real,3>(left, bottom, near);
	return mat<real,4,4>{
		{near2 / diff.x, 0, (right + left) / diff.x, 0},
		{0, near2 / diff.y, (top + bottom) / diff.y, 0},
		{0, 0, -(near + far) / diff.z, -(near2 * far) / diff.z},
		{0, 0, -1, 0},
	};
}

/*
gluPerspective
http://www.songho.ca/opengl/gl_transform.html
https://www.khronos.org/opengl/wiki/GluPerspective_code
TODO tempted to make an optimized storage class for this as well.
It's *almost* a diagonal scale class (min(m,n) reals for m x n matrix) 
though this have to be specialized for perspective since it has that one off-diagonal element...
I see this as the library slowly creeping into sparse-matrix storage. 
*/
template<typename real>
mat<real,4,4> perspective(
	real fovY,	// in radians
	real aspectRatio,
	real near,
	real far
) {
	auto ymax = near * tan(fovY * (real).5);
	auto xmax = aspectRatio * ymax;
	return frustum(-xmax, xmax, -ymax, ymax, near, far);
}

// glOrtho
// https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/orthographic-projection-matrix.html
template<typename real>
mat<real,4,4> ortho(
	real left,
	real right,
	real bottom,
	real top,
	real near,
	real far
) {
	auto diff = vec<real,3>(right, top, far) - vec<real,3>(left, bottom, near);
	return mat<real,4,4>{
		{2 / diff.x, 0, 0, -(right + left) / diff.x},
		{0, 2 / diff.y, 0, -(top + bottom) / diff.y},
		{0, 0, -2 / diff.z, -(near + far) / diff.z},
		{0, 0, 0, 1},
	};
}

// gluOrtho2D
template<typename real>
mat<real,4,4> ortho2D(
	real left,
	real right,
	real bottom,
	real top
) {
	return ortho(left, right, bottom, top, -1, 1);
}

}
