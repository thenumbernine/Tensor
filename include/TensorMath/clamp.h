#pragma once

template<typename T>
T& clamp(T &x, T &min, T &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

template<typename T>
const T& clamp(const T &x, const T &min, const T &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

