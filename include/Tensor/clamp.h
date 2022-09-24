#pragma once

template<typename T>
T& clamp(T &x, T &min, T &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

template<typename T>
T const& clamp(T const &x, T const &min, T const &max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}
