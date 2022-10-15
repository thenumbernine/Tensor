#pragma once

namespace Tensor {

//index-access classes
struct IndexBase;

template<char ident>
struct Index;

//forward-declare for index-access
template<typename Tensor_, typename IndexVector>
struct IndexAccess;

}
