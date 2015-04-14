﻿//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
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

#light

/// Non-nested 1st-order forward mode AD
namespace DiffSharp.AD.Specialized.Forward1

open DiffSharp.Util.General
open FsAlg.Generic

/// Dual numeric type, keeping primal and tangent values
[<CustomEquality; CustomComparison>]
type Dual =
    // Primal, tangent
    | Dual of float * float
    override d.ToString() = let (Dual(p, t)) = d in sprintf "Dual(%A, %A)" p t
    static member op_Explicit(p) = Dual(p, 0.)
    static member op_Explicit(Dual(p, _)) = p
    static member DivideByInt(Dual(p, t), i:int) = Dual(p / float i, t / float i)
    static member Zero = Dual(0., 0.)
    static member One = Dual(1., 0.)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? Dual as d2 -> let Dual(a, _), Dual(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this Dual with another type of object."
    override d.Equals(other) = 
        match other with
        | :? Dual as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (Dual(a, b)) = d in hash [|a; b|]
    // Dual - Dual binary operations
    static member (+) (Dual(a, at), Dual(b, bt)) = Dual(a + b, at + bt)
    static member (-) (Dual(a, at), Dual(b, bt)) = Dual(a - b, at - bt)
    static member (*) (Dual(a, at), Dual(b, bt)) = Dual(a * b, at * b + a * bt)
    static member (/) (Dual(a, at), Dual(b, bt)) = Dual(a / b, (at * b - a * bt) / (b * b))
    static member Pow (Dual(a, at), Dual(b, bt)) = let apb = a ** b in Dual(apb, apb * ((b * at / a) + ((log a) * bt)))
    static member Atan2 (Dual(a, at), Dual(b, bt)) = Dual(atan2 a b, (at * b - a * bt) / (a * a + b * b))
    // Dual - float binary operations
    static member (+) (Dual(a, at), b) = Dual(a + b, at)
    static member (-) (Dual(a, at), b) = Dual(a - b, at)
    static member (*) (Dual(a, at), b) = Dual(a * b, at * b)
    static member (/) (Dual(a, at), b) = Dual(a / b, at / b)
    static member Pow (Dual(a, at), b) = Dual(a ** b, b * (a ** (b - 1.)) * at)
    static member Atan2 (Dual(a, at), b) = Dual(atan2 a b, (b * at) / (b * b + a * a))
    // float - Dual binary operations
    static member (+) (a, Dual(b, bt)) = Dual(b + a, bt)
    static member (-) (a, Dual(b, bt)) = Dual(a - b, -bt)
    static member (*) (a, Dual(b, bt)) = Dual(b * a, bt * a)
    static member (/) (a, Dual(b, bt)) = Dual(a / b, -a * bt / (b * b))
    static member Pow (a, Dual(b, bt)) = let apb = a ** b in Dual(apb, apb * (log a) * bt)
    static member Atan2 (a, Dual(b, bt)) = Dual(atan2 a b, -(a * bt) / (a * a + b * b))
    // Dual - int binary operations
    static member (+) (a:Dual, b:int) = a + float b
    static member (-) (a:Dual, b:int) = a - float b
    static member (*) (a:Dual, b:int) = a * float b
    static member (/) (a:Dual, b:int) = a / float b
    static member Pow (a:Dual, b:int) = Dual.Pow(a, float b)
    static member Atan2 (a:Dual, b:int) = Dual.Atan2(a, float b)
    // int - Dual binary operations
    static member (+) (a:int, b:Dual) = (float a) + b
    static member (-) (a:int, b:Dual) = (float a) - b
    static member (*) (a:int, b:Dual) = (float a) * b
    static member (/) (a:int, b:Dual) = (float a) / b
    static member Pow (a:int, b:Dual) = Dual.Pow(float a, b)
    static member Atan2 (a:int, b:Dual) = Dual.Atan2(float a, b)
    // Dual unary operations
    static member Log (Dual(a, at)) =
        if a <= 0. then invalidArgLog()
        Dual(log a, at / a)
    static member Log10 (Dual(a, at)) =
        if a <= 0. then invalidArgLog10()
        Dual(log10 a, at / (a * log10val))
    static member Exp (Dual(a, at)) = let expa = exp a in Dual(expa, at * expa)
    static member Sin (Dual(a, at)) = Dual(sin a, at * cos a)
    static member Cos (Dual(a, at)) = Dual(cos a, -at * sin a)
    static member Tan (Dual(a, at)) =
        let cosa = cos a
        if cosa = 0. then invalidArgTan()
        Dual(tan a, at / (cosa * cosa))
    static member (~-) (Dual(a, at)) = Dual(-a, -at)
    static member Sqrt (Dual(a, at)) = 
        if a <= 0. then invalidArgSqrt()
        let sqrta = sqrt a in Dual(sqrta, at / (2. * sqrta))
    static member Sinh (Dual(a, at)) = Dual(sinh a, at * cosh a)
    static member Cosh (Dual(a, at)) = Dual(cosh a, at * sinh a)
    static member Tanh (Dual(a, at)) = let cosha = cosh a in Dual(tanh a, at / (cosha * cosha))
    static member Asin (Dual(a, at)) =
        if (abs a) >= 1. then invalidArgAsin()
        Dual(asin a, at / sqrt (1. - a * a))
    static member Acos (Dual(a, at)) = 
        if (abs a) >= 1. then invalidArgAcos()
        Dual(acos a, -at / sqrt (1. - a * a))
    static member Atan (Dual(a, at)) = Dual(atan a, at / (1. + a * a))
    static member Abs (Dual(a, at)) = 
        if a = 0. then invalidArgAbs()
        Dual(abs a, at * float (sign a))
    static member Floor (Dual(a, _)) =
        if isInteger a then invalidArgFloor()
        Dual(floor a, 0.)
    static member Ceiling (Dual(a, _)) =
        if isInteger a then invalidArgCeil()
        Dual(ceil a, 0.)
    static member Round (Dual(a, _)) =
        if isHalfway a then invalidArgRound()
        Dual(round a, 0.)

/// Dual operations module (automatically opened)
[<AutoOpen>]
module DualOps =
    /// Make Dual, with primal value `p` and tangent 0
    let inline dual p = Dual(float p, 0.)
    /// Make Dual, with primal value `p` and tangent value `t`
    let inline dualPT p t = Dual(float p, float t)
    /// Make active Dual (i.e. variable of differentiation), with primal value `p` and tangent 1
    let inline dualP1 p = Dual(float p, 1.)
    /// Get the primal value of a Dual
    let inline primal (Dual(p, _)) = p
    /// Get the tangent value of a Dual
    let inline tangent (Dual(_, t)) = t
    /// Get the primal and tangent values of a Dual, as a tuple
    let inline tuple (Dual(p, t)) = (p, t)


/// Forward differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        x |> dualP1 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f (x:float) =
        x |> dualP1 |> f |> tangent
       
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' f (x:float[]) (v:float[]) =
        Array.map2 dualPT x v |> f |> tuple

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv f x v =
        gradv' f x v |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:float[]) =
        let a = Array.init x.Length (fun i -> gradv' f x (standardBasis x.Length i))
        (fst a.[0], Array.map snd a)

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' f (x:float[]) (v:float[]) = 
        Array.map2 dualPT x v |> f |> Array.map tuple |> Array.unzip

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv f (x:float[]) (v:float[]) = 
        jacobianv' f x v |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f (x:float[]) =
        let a = Array.init x.Length (fun i -> jacobianv' f x (standardBasis x.Length i))
        (fst a.[0], array2D (Array.map snd a))

    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f x =
        jacobianT' f x |> fun (r, j) -> (r, transpose j)

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        curl' f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        v, trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        div' f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        curldiv' f x |> sndtrd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:Dual->Dual) x = DiffOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:Dual->Dual) x = DiffOps.diff f x
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv' (f:Vector<Dual>->Dual) x v = DiffOps.gradv' (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    let inline gradv (f:Vector<Dual>->Dual) x v = DiffOps.gradv (vector >> f) (Vector.toArray x) (Vector.toArray v)
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<Dual>->Dual) x = DiffOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<Dual>->Dual) x = DiffOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<Dual>->Vector<_>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<Dual>->Vector<_>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<Dual>->Vector<_>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<Dual>->Vector<_>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv' (f:Vector<Dual>->Vector<Dual>) x v = DiffOps.jacobianv' (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> fun (a, b) -> (vector a, vector b)
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    let inline jacobianv (f:Vector<Dual>->Vector<Dual>) x v = DiffOps.jacobianv (vector >> f >> Vector.toArray) (Vector.toArray x) (Vector.toArray v) |> vector
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<Dual>->Vector<Dual>) x = DiffOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<Dual>->Vector<Dual>) x = DiffOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<Dual>->Vector<Dual>) x = DiffOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<Dual>->Vector<Dual>) x = DiffOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<Dual>->Vector<Dual>) x = DiffOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<Dual>->Vector<Dual>) x = DiffOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
