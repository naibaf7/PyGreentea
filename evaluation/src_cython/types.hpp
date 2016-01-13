#pragma once

#include <boost/multi_array.hpp>
#include <boost/multi_array/types.hpp>

#include <memory>
#include <vector>
#include <tuple>

template < typename T > struct watershed_traits;

template <> struct watershed_traits<uint32_t>
{
    static const uint32_t high_bit = 0x80000000;
    static const uint32_t mask     = 0x7FFFFFFF;
    static const uint32_t visited  = 0x00001000;
    static const uint32_t dir_mask = 0x0000007F;
};

template <> struct watershed_traits<uint64_t>
{
    static const uint64_t high_bit = 0x8000000000000000LL;
    static const uint64_t mask     = 0x7FFFFFFFFFFFFFFFLL;
    static const uint64_t visited  = 0x0000000000001000LL;
    static const uint64_t dir_mask = 0x000000000000007FLL;
};

template < typename T >
using volume = boost::multi_array<T,3>;

template < typename T >
using affinity_graph = boost::multi_array<T,4>;

template < typename T >
using volume_ptr = std::shared_ptr<volume<T>>;

template < typename T >
using affinity_graph_ptr = std::shared_ptr<affinity_graph<T>>;

template< typename ID, typename F >
using region_graph = std::vector<std::tuple<F,ID,ID>>;

template< typename ID, typename F >
using region_graph_ptr = std::shared_ptr<region_graph<ID,F>>;
