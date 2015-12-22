//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under the LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

namespace DiffSharp.Config

open DiffSharp.Backend

/// Record type holding configuration parameters
type Config =
    {floatTypeBackend : cudaBackend
     floatTypeEpsilon : floatType
     floatTypeEpsilonRec : floatType
     floatTypeEpsilonRec2 : floatType
     floatTypeVisualizationContrast : floatType
     GrayscalePalette : string[]}

/// Global configuration
type GlobalConfig() =
    static let GrayscalePaletteUnicode = [|" "; "·"; "-"; "▴"; "▪"; "●"; "♦"; "■"; "█"|]
    static let GrayscalePaletteASCII = [|" "; "."; ":"; "x"; "T"; "Y"; "V"; "X"; "H"; "N"; "M"|]
    static let mutable C =
        let eps = floatType 0.00001
        {floatTypeBackend = cudaBackend()
         floatTypeEpsilon = eps
         floatTypeEpsilonRec = (floatType 1.0) / eps
         floatTypeEpsilonRec2 = (floatType 0.5) / eps
         floatTypeVisualizationContrast = floatType 1.2
         GrayscalePalette = GrayscalePaletteUnicode}

    static member floatTypeBackend = C.floatTypeBackend
    static member floatTypeEpsilon = C.floatTypeEpsilon
    static member floatTypeEpsilonRec = C.floatTypeEpsilonRec
    static member floatTypeEpsilonRec2 = C.floatTypeEpsilonRec2
    static member floatTypeVisualizationContrast = C.floatTypeVisualizationContrast
    static member GrayscalePalette = C.GrayscalePalette
    static member SetEpsilon(e:floatType) = 
        C <- {C with
                floatTypeEpsilon = e
                floatTypeEpsilonRec = (floatType 1.) / e
                floatTypeEpsilonRec2 = (floatType 0.5) / e}
    static member SetVisualizationContrast(c:floatType) =
        C <- {C with
                floatTypeVisualizationContrast = c}
    static member SetVisualizationPalette(palette:string) =
        match palette with
        | "ASCII" ->
            C <- {C with
                    GrayscalePalette = GrayscalePaletteASCII}
        | "Unicode" ->
            C <- {C with
                    GrayscalePalette = GrayscalePaletteUnicode}
        | _ -> invalidArg "" "Unsupported palette. Try: ASCII or Unicode"