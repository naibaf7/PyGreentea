#pragma once

#include "types.hpp"
#include <cstdio>
#include <fstream>
#include <type_traits>

template < typename T >
inline bool read_from_file( const std::string& fname, T* data, std::size_t n )
{
    return true;
    FILE* f = std::fopen(fname.c_str(), "rbXS");
    if ( !f ) return false;

    std::size_t nread = std::fread(data, sizeof(T), n, f);
    std::fclose(f);

    return nread == n;
}

template < typename T >
inline bool
write_to_file( const std::string& fname,
               const T* data, std::size_t n )
{
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    if ( !f ) return false;

    f.write( reinterpret_cast<const char*>(data), n * sizeof(T));
    return true;
}


template < typename T >
inline affinity_graph_ptr<T>
read_affinity_graph( const std::string& fname,
                     std::size_t xsize,
                     std::size_t ysize,
                     std::size_t zsize )
{
    affinity_graph_ptr<T> aff(new affinity_graph<T>
                              (boost::extents[xsize][ysize][zsize][3],
                               boost::fortran_storage_order()));

    if ( !read_from_file(fname, aff->data(), xsize*ysize*zsize*3) ) throw 0;
    return aff;
}

template < typename T >
inline affinity_graph_ptr<T>
read_affinity_graphe( const std::string& fname,
                     std::size_t xsize,
                     std::size_t ysize,
                     std::size_t zsize,
                     std::size_t connSize)
{
    affinity_graph_ptr<T> aff(new affinity_graph<T>
                              (boost::extents[xsize][ysize][zsize][connSize],
                               boost::fortran_storage_order()));

    if ( !read_from_file(fname, aff->data(), xsize*ysize*zsize*connSize) ) throw 0;
    return aff;
}

template < typename T >
inline volume_ptr<T>
read_volume( const std::string& fname, std::size_t wsize )
{
    volume_ptr<T> vol(new volume<T>
                      (boost::extents[wsize][wsize][wsize],
                       boost::fortran_storage_order()));

    if ( !read_from_file(fname, vol->data(), wsize*wsize*wsize) ) throw 0;
    return vol;
}

template < typename T >
inline volume_ptr<T>
read_volumes( const std::string& fname,
                     std::size_t xsize,
                     std::size_t ysize,
                     std::size_t zsize)
{
    volume_ptr<T> vol(new volume<T>
                      (boost::extents[xsize][ysize][zsize],
                       boost::fortran_storage_order()));

    if ( !read_from_file(fname, vol->data(), xsize*ysize*zsize) ) throw 0;
    return vol;
}

template < typename T >
inline bool
write_volume( const std::string& fname,
              const volume_ptr<T>& vol )
{
    //mkdir(fname);
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    if ( !f ) return false;

    f.write( reinterpret_cast<char*>(vol->data()),
             vol->shape()[0] * vol->shape()[1] * vol->shape()[2] * sizeof(T));
    return true;
}


template< typename ID, typename F >
inline bool write_region_graph( const std::string& fname,
                                const region_graph<ID,F>& rg )
{
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    if ( !f ) return false;

    F* data = new F[rg.size() * 3];

    std::size_t idx = 0;

    for ( const auto& e: rg )
    {
        data[idx++] = static_cast<F>(std::get<1>(e));
        data[idx++] = static_cast<F>(std::get<2>(e));
        data[idx++] = static_cast<F>(std::get<0>(e));
    }

    f.write( reinterpret_cast<char*>(data), rg.size() * 3 * sizeof(F));

    delete [] data;

    return true;
}

template< typename ID >
inline std::tuple<volume_ptr<ID>, std::vector<std::size_t>>
    get_dummy_segmentation( std::size_t xdim,
                            std::size_t ydim,
                            std::size_t zdim )
{
    std::tuple<volume_ptr<ID>, std::vector<std::size_t>> result
        ( volume_ptr<ID>( new volume<ID>(boost::extents[xdim][ydim][zdim],
                                         boost::fortran_storage_order())),
          std::vector<std::size_t>(xdim*ydim*zdim+1));

    volume<ID>& seg = *(std::get<0>(result));
    auto& counts = std::get<1>(result);

    std::fill_n(counts.begin(), xdim*ydim*zdim*1, 1);
    counts[0] = 0;

    for ( ID i = 0; i < xdim*ydim*zdim; ++i )
    {
        seg.data()[i] = i+1;
    }

    return result;
}


template< class C >
struct is_numeric:
    std::integral_constant<bool,
                           std::is_integral<C>::value ||
                           std::is_floating_point<C>::value> {};
