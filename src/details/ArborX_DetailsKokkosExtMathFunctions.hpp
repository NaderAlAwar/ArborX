/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace KokkosExt
{

#if KOKKOS_VERSION >= 30699
using Kokkos::isfinite;
#else
using Kokkos::Experimental::isfinite;
#endif

} // namespace KokkosExt

#endif
