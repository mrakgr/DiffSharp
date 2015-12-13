﻿#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\ManagedCuda.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\NVRTC.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\CudaBlas.dll"
#r @"C:\Users\Marko\documents\visual studio 2015\Projects\Automatic Differentiation\packages\ManagedCuda-75-x64.7.5.7\lib\net45\x64\CudaSolve.dll"

open ManagedCuda
open ManagedCuda.VectorTypes
open ManagedCuda.BasicTypes
open ManagedCuda.NVRTC
open ManagedCuda.CudaBlas

let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_dev' (host_ar: 't [,]) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    h_a

let new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

let ctx = new CudaContext()

// TODO: Alea binds cuBLAS and cuDNN contexts to a worker. Need to figure out how that works before calling it done.
//let str = CUstream()

/// The cuBlas handle.
let c = CudaBlas()

let rng = System.Random()

let isamax(x:float32[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.Max(d_x,arg_incx)

// y <- alpha * x + y
let saxpy(alpha:float32, x:float32[], y:float32[]) =
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_incx = 1
    let arg_incy = 1
    c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
    d_y.CopyToHost(y)

// Y <- alpha * X + Y
let saxpy'(alpha:float32, x:float32[,], y:float32[,]) =
    use d_x = to_dev' x
    use d_y = to_dev' y
    let arg_incx = 1
    let arg_incy = 1
    c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
    d_y.CopyToHost(y)

let sscal(alpha: float32, x: float32[]) =
    use d_x = to_dev x
    let arg_incx = 1
    let arg_incy = 1
    c.Scale(alpha,d_x,arg_incx)
    d_x.CopyToHost(x)

let sscal'(alpha: float32, x: float32[,]) =
    use d_x = to_dev' x
    let arg_incx = 1
    c.Scale(alpha,d_x,arg_incx)
    d_x.CopyToHost(x)

let sdot(x: float32[], y: float32[]) =
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_incx = 1
    let arg_incy = 1
    c.Dot(d_x,arg_incx,d_y,arg_incy)

let sger(alpha: float32, x:float32[], y:float32[], a:float32[,]) =
    use d_x = to_dev x
    use d_y = to_dev y
    use d_a = to_dev' a
    let arg_m = y.Length
    let arg_n = x.Length
    let arg_alpha = alpha
    let arg_incx = 1
    let arg_incy = 1
    let arg_lda = arg_m

    // As it happens, the wrapped c.Ger does not allow reversing the m and n arguments, so I have taken it out here.
    // This is an adjustment so it works for row major matrices.
    let _blasHandle = c.CublasHandle
    let _status = CudaBlasNativeMethods.cublasSger_v2(_blasHandle, arg_m, arg_n, ref alpha, d_x.DevicePointer, arg_incx, d_y.DevicePointer, arg_incy, d_a.DevicePointer, arg_lda)
    if (_status <> CublasStatus.Success) then raise (new CudaBlasException(_status))
    d_a.CopyToHost(a)

let sasum(x:float32[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.AbsoluteSum(d_x,arg_incx)

let snrm2(x:float32[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.Norm2(d_x,arg_incx)

// O <- alpha * A * B + beta * O
let sgemm(alpha:float32, a:float32[,], b:float32[,], beta:float32, o:float32[,]) =
    let d_a = to_dev' a
    let d_b = to_dev' b
    let d_o = to_dev' o
    // Order modified to work with row-major matrices and eliminate the need for transposing the result
    let m = Array2D.length1 a
    let n = Array2D.length2 b
    let k = Array2D.length1 b
    let arg_transa = Operation.NonTranspose
    let arg_transb = Operation.NonTranspose
    let arg_m = n
    let arg_n = m
    let arg_k = k
    let arg_alpha = alpha
    let arg_lda = n
    let arg_ldb = k
    let arg_beta = beta
    let arg_ldc = n
    c.Gemm(arg_transa,arg_transb,arg_m, arg_n, arg_k, arg_alpha, d_a, arg_lda, d_b, arg_ldb, arg_beta, d_o, arg_ldc)
    d_o.CopyToHost(o)

// y <- alpha * A * x + beta * y
let sgemv(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
    use d_a = to_dev' a
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_trans = Operation.Transpose
    let arg_m = Array2D.length2 a
    let arg_n = Array2D.length1 a
    let arg_alpha = alpha
    let arg_lda = arg_m
    let arg_incx = 1
    let arg_beta = beta
    let arg_incy = 1
    c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
    d_y.CopyToHost(y)

// y <- alpha * x * A + beta * y
let sgemv'(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
    use d_a = to_dev' a
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_trans = Operation.NonTranspose
    let arg_m = Array2D.length2 a
    let arg_n = Array2D.length1 a
    let arg_alpha = alpha
    let arg_lda = arg_m
    let arg_incx = 1
    let arg_beta = beta
    let arg_incy = 1
    c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
    d_y.CopyToHost(y)


let idamax(x:float[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.Max(d_x,arg_incx)

// y <- alpha * x + y
let daxpy(alpha:float, x:float[], y:float[]) =
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_incx = 1
    let arg_incy = 1
    c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
    d_y.CopyToHost(y)

// Y <- alpha * X + Y
let daxpy'(alpha:float, x:float[,], y:float[,]) =
    use d_x = to_dev' x
    use d_y = to_dev' y
    let arg_incx = 1
    let arg_incy = 1
    c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
    d_y.CopyToHost(y)

let dscal(alpha: float, x: float[]) =
    use d_x = to_dev x
    let arg_incx = 1
    let arg_incy = 1
    c.Scale(alpha,d_x,arg_incx)
    d_x.CopyToHost(x)

let dscal'(alpha: float, x: float[,]) =
    use d_x = to_dev' x
    let arg_incx = 1
    c.Scale(alpha,d_x,arg_incx)
    d_x.CopyToHost(x)

let ddot(x: float[], y: float[]) =
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_incx = 1
    let arg_incy = 1
    c.Dot(d_x,arg_incx,d_y,arg_incy)

let dger(alpha: float, x:float[], y:float[], a:float[,]) =
    use d_x = to_dev x
    use d_y = to_dev y
    use d_a = to_dev' a
    let arg_m = y.Length
    let arg_n = x.Length
    let arg_alpha = alpha
    let arg_incx = 1
    let arg_incy = 1
    let arg_lda = arg_m

    // As it happens, the wrapped c.Ger does not allow reversing the m and n arguments, so I have taken it out here.
    // This is an adjustment so it works for row major matrices.
    let _blasHandle = c.CublasHandle
    let _status = CudaBlasNativeMethods.cublasDger_v2(_blasHandle, arg_m, arg_n, ref alpha, d_x.DevicePointer, arg_incx, d_y.DevicePointer, arg_incy, d_a.DevicePointer, arg_lda)
    if (_status <> CublasStatus.Success) then raise (new CudaBlasException(_status))
    d_a.CopyToHost(a)

let dasum(x:float[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.AbsoluteSum(d_x,arg_incx)

let dnrm2(x:float[]) =
    use d_x = to_dev x
    let arg_incx = 1
    c.Norm2(d_x,arg_incx)

// O <- alpha * A * B + beta * O
let dgemm(alpha:float, a:float[,], b:float[,], beta:float, o:float[,]) =
    let d_a = to_dev' a
    let d_b = to_dev' b
    let d_o = to_dev' o
    // Order modified to work with row-major matrices and eliminate the need for transposing the result
    let m = Array2D.length1 a
    let n = Array2D.length2 b
    let k = Array2D.length1 b
    let arg_transa = Operation.NonTranspose
    let arg_transb = Operation.NonTranspose
    let arg_m = n
    let arg_n = m
    let arg_k = k
    let arg_alpha = alpha
    let arg_lda = n
    let arg_ldb = k
    let arg_beta = beta
    let arg_ldc = n
    c.Gemm(arg_transa,arg_transb,arg_m, arg_n, arg_k, arg_alpha, d_a, arg_lda, d_b, arg_ldb, arg_beta, d_o, arg_ldc)
    d_o.CopyToHost(o)

// y <- alpha * A * x + beta * y
let dgemv(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
    use d_a = to_dev' a
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_trans = Operation.Transpose
    let arg_m = Array2D.length2 a
    let arg_n = Array2D.length1 a
    let arg_alpha = alpha
    let arg_lda = arg_m
    let arg_incx = 1
    let arg_beta = beta
    let arg_incy = 1
    c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
    d_y.CopyToHost(y)

// y <- alpha * x * A + beta * y
let dgemv'(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
    use d_a = to_dev' a
    use d_x = to_dev x
    use d_y = to_dev y
    let arg_trans = Operation.NonTranspose
    let arg_m = Array2D.length2 a
    let arg_n = Array2D.length1 a
    let arg_alpha = alpha
    let arg_lda = arg_m
    let arg_incx = 1
    let arg_beta = beta
    let arg_incy = 1
    c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
    d_y.CopyToHost(y)

let s = new CudaSolve.CudaSolveDense()

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security
open System.Threading.Tasks

[<SuppressUnmanagedCodeSecurity>]
[<DllImport(@"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\DiffSharp.0.7.5\build\libopenblas.dll", EntryPoint="sgesv_")>]
extern void sgesv_(int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, int *info)

[<SuppressUnmanagedCodeSecurity>]
[<DllImport(@"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\DiffSharp.0.7.5\build\libopenblas.dll", EntryPoint="ssysv_")>]
extern void ssysv_(char *uplo, int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, float32 *work, int *lwork, int *info)

[<SuppressUnmanagedCodeSecurity>]
[<DllImport(@"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\DiffSharp.0.7.5\build\libopenblas.dll", EntryPoint="sgetrf_")>]
extern void sgetrf_(int *m, int *n, float32 *a, int *lda, int *ipiv, int *info)

[<SuppressUnmanagedCodeSecurity>]
[<DllImport(@"C:\Users\Marko\Documents\Visual Studio 2015\Projects\Automatic Differentiation\packages\DiffSharp.0.7.5\build\libopenblas.dll", EntryPoint="sgetri_")>]
extern void sgetri_(int *n, float32 *a, int *lda, int *ipiv, float32 *work, int *lwork, int *info)

#nowarn "9"
#nowarn "51"

let parallelizationThreshold = 50000
type PinnedArray<'T when 'T : unmanaged> (array : 'T[]) =
    let h = GCHandle.Alloc(array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose() = h.Free()

type PinnedArray2D<'T when 'T : unmanaged> (array : 'T[,]) =
    let h = GCHandle.Alloc(array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose() = h.Free()

// Compared to the LAPACK sgesv, there are slight numerical deviations on the order of 1e-4 relative to the size of the inputs for this function.
// To get the equivalent of sgesv, getrf (LU factorization) and getrs (solver) have to be called seperately.

// There is also a call to geam (matrix matrix add function) purely for the sake of transposition to column major format.
// It might be worth considering going to column major fully as both OpenBlas, Lapack and now these Cuda library functions use column major natively.
let sgesv(a:float32[,], b:float32[]) =
    let m = Array2D.length1 a
    let n = Array2D.length2 a

    let b' = Array.copy b

    use d_b = to_dev b
    let ipiv = Array.zeroCreate n
    use d_ipiv = new_dev<int> n

    let arg_n = n
    let arg_nrhs = 1
    let arg_lda = n
    let arg_ldb = n
    
    use d_nta = to_dev' a
    use d_a = new_dev<float32> a.Length
    c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

    let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
    use workspace = new_dev<float32> Lwork
    
    use d_info = to_dev [|0|]
    s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

    let factorization_par = d_info.[SizeT 0]
    if factorization_par <> 0 then failwithf "Parameter %i in sgesv is incorrect." factorization_par
    
    s.Getrs(Operation.NonTranspose,arg_n,arg_nrhs,d_a,arg_lda,d_ipiv,d_b,arg_ldb,d_info)
    d_b.CopyToHost(b')
    
    if d_info.[SizeT 0] = 0 then
        Some(b')
    else
        None

let ssysv(a:float32[,], b:float32[]) =
    let n = Array2D.length1 a
    let ipiv = Array.zeroCreate<float32> n
    let work = Array.zeroCreate<float32> 1

    use d_nta = to_dev' a
    use d_a = new_dev<float32> a.Length
    c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

    use d_ipiv = new_dev<int> n
    use d_b = to_dev b

    let arg_n = n
    let arg_nrhs = 1
    let arg_lda = n
    let arg_ldb = n

    let Lwork = s.PotrfBufferSize(FillMode.Upper,n,d_nta,arg_lda)
    use d_work = new_dev<float32> Lwork
    use d_info = to_dev [|0|]

    s.Potrf(FillMode.Upper,arg_n,d_a,arg_lda,d_work,Lwork,d_info)

    let factorization_par = d_info.[SizeT 0]
    if factorization_par <> 0 then failwithf "Parameter %i in ssysv is incorrect." factorization_par

    s.Potrs(FillMode.Upper,arg_n,arg_nrhs,d_a,arg_lda,d_b,arg_ldb,d_info)

    if d_info.[SizeT 0] = 0 then
        Some(to_host d_b)
    else
        None

let sgetrf(a:float32[,]) =
    let m = Array2D.length1 a
    let n = Array2D.length2 a

    use d_ipiv = new_dev<int> (min m n)

    let m = Array2D.length1 a
    let n = Array2D.length2 a
    let arg_m = m
    let arg_n = n
    let arg_lda = m
    
    use d_nta = to_dev' a
    use d_a = new_dev<float32> a.Length
    c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

    let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
    use workspace = new_dev<float32> Lwork
    
    use d_info = to_dev [|0|]
    s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

    if d_info.[SizeT 0] = 0 then
        Some(to_host d_ipiv)
    else
        None

// The MKL Lapack manual states that the sgetri (matrix inversion) function requires sgetrf to factorize the matrix before being called.
// Maybe it would be good for this one to replace sgetrf as it returns both the LU and the pivots matrices.
let sgetrf2(a:float32[,]) =
    let m = Array2D.length1 a
    let n = Array2D.length2 a

    use d_ipiv = new_dev<int> (min m n)

    let m = Array2D.length1 a
    let n = Array2D.length2 a
    let arg_m = m
    let arg_n = n
    let arg_lda = m
    
    use d_nta = to_dev' a
    use d_a = new_dev<float32> a.Length
    c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

    let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
    use workspace = new_dev<float32> Lwork
    
    use d_info = to_dev [|0|]
    s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

    if d_info.[SizeT 0] = 0 then
        let h_a = Array2D.zeroCreate<float32> m n
        d_a.CopyToHost(h_a)
        Some(h_a,to_host d_ipiv)
    else
        None

// Strangely enough, cuSolver does not have a matrix inverse, but cuBlas does.
// Has no transpose step as I assume it is intended to be called after sgetrf.

// Does this function intend to mutate a and ipiv?
// Given that it returns a value I am assuming that it does not.
let sgetri(a:float32[,], ipiv:int[]) =
    use d_a = to_dev' a
    use d_ar_a = new_dev<CUdeviceptr> 1
    d_ar_a.[SizeT 0] <- d_a.DevicePointer
    use d_ipiv = to_dev ipiv

    let n = Array2D.length1 a
    use d_work = new_dev<float32> (n * n)
    use d_ar_work = new_dev<CUdeviceptr> 1
    d_ar_work.[SizeT 0] <- d_work.DevicePointer

    let arg_n = n
    let arg_lda = n
    let arg_ldc = n
    let arg_lwork = n * n
    use d_info = to_dev [|0|]
    c.GetriBatchedS(arg_n,d_ar_a,arg_lda,d_ipiv,d_ar_work,arg_ldc,d_info,1)
    if d_info.[SizeT 0] = 0 then
        Some(to_host d_work)
    else
        None

let n = 50
let a' =   [|
        [| 1.0f; -1.0f; -1.0f; -1.0f; -1.0f; |];
        [| -1.0f; 2.0f; 0.0f; 0.0f; 0.0f; |];
        [| -1.0f; 0.0f; 3.0f; 1.0f; 1.0f; |];
        [| -1.0f; 0.0f; 1.0f; 4.0f; 2.0f; |];
        [| -1.0f; 0.0f; 1.0f; 2.0f; 5.0f; |]
          |]
//let a = Array2D.init n n (fun x y -> a'.[x].[y])
let a = Array2D.init n n (fun _ _ -> (rng.NextDouble()-0.5)*6.0 |> float32)
let b = Array.init n (fun _ -> (rng.NextDouble()-0.5)*6.0 |> float32)

let r0 = sgesv(a,b)
let r1r = sgesv'(a,b)
