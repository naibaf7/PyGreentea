//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZI_BITS_COMPLEX_HPP
#define ZI_BITS_COMPLEX_HPP 1

#include <zi/config/config.hpp>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#  include <complex>
#  define ZI_COMPLEX_NAMESPACE ::std
#else
#  if defined( ZI_USE_TR1 ) || defined( ZI_NO_BOOST )
#    include <tr1/complex>
#    define ZI_COMPLEX_NAMESPACE ::std::tr1
#  else
#    include <boost/tr1/complex.hpp>
#    define ZI_COMPLEX_NAMESPACE ::std::tr1
#  endif
#endif

namespace zi {

using ZI_COMPLEX_NAMESPACE::acos;
using ZI_COMPLEX_NAMESPACE::asin;
using ZI_COMPLEX_NAMESPACE::atan;
using ZI_COMPLEX_NAMESPACE::acosh;
using ZI_COMPLEX_NAMESPACE::asinh;
using ZI_COMPLEX_NAMESPACE::atanh;
using ZI_COMPLEX_NAMESPACE::fabs;

using ::std::atan2;
using ::std::arg;
using ::std::norm;
using ::std::conj;
using ::std::polar;
using ::std::imag;
using ::std::real;
using ::std::pow;

using ::std::complex;

} // namespace zi

#undef ZI_COMPLEX_NAMESPACE
#endif
