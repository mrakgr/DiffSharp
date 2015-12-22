#nowarn "9"
#nowarn "51"

namespace DiffSharp.Backend

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security

open ManagedCuda
open ManagedCuda.VectorTypes
open ManagedCuda.BasicTypes
open ManagedCuda.NVRTC
open ManagedCuda.CudaBlas

[<AutoOpen>]
module Cuda =
    // The backend can be switched between float32 and float by modifying the type alias below...if it was not for the Cuda math functions.
    // They would also have to be renamed manually to their double names.
    type floatType = float32
    let inline floatType x = float32 x
    [<Literal>]
    let FloatTypeCpp = "float"
    /// The global variable that sets bounds checking.
    [<Literal>]
    let DoBoundsChecking = true

    /// The cuBlas handle.
    let ctx = new CudaContext()
    let str = new CudaStream() // The stream makes sure that all the kernels are scheduled correctly. In Cuda they are lauched asynchronously.
    let c = CudaBlas(str.Stream)
    let s = new CudaSolve.CudaSolveDense(str)

    let inline to_dev (host_ar: 't []) =
        let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
        d_a.CopyToDevice(host_ar)
        d_a

    let to_dev' (host_ar: 't [,]) =
        let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
        d_a.CopyToDevice(host_ar)
        d_a

    let inline to_host<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (dev_ar: CudaDeviceVariable<'t>) =
        let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
        dev_ar.CopyToHost(h_a)
        h_a

    type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
        member inline this.Gather() =
            to_host this

    let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
        new CudaDeviceVariable<'t>(SizeT n)

    type dMatrix(num_rows:int,num_cols:int,dArray: CudaDeviceVariable<floatType>) = 
            new(num_rows: int,num_cols) =
                let q = (num_rows*num_cols) |> SizeT
                let t = new CudaDeviceVariable<floatType>(q)
                new dMatrix(num_rows,num_cols,t)

            new(num_rows: int,num_cols,dArray: floatType[]) =
                let q = num_rows*num_cols
                if dArray.Length <> q then failwith "Invalid size in dMatrix construction."
                let t = to_dev dArray
                new dMatrix(num_rows,num_cols,t)

            member t.num_rows = num_rows
            member t.num_cols = num_cols
            member t.dArray = dArray

            member t.rc = t.num_rows, t.num_cols
            /// Sets the matrix to zero.
            member t.setZero() = t.dArray.MemsetAsync(0u,str.Stream)
            /// Set the matrix to a value.
            member t.set (x: floatType) = 
                let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
                t.dArray.MemsetAsync(v,str.Stream)
            /// Creates a copy of this matrix with all the values set to zero.
            member t.zeroLike() =
                let c = new dMatrix(t.num_rows,t.num_cols)
                c.setZero()
                c
            member t.copy() =
                let c = new dMatrix(t.num_rows,t.num_cols)
                c.dArray.AsyncCopyToDevice(t.dArray,str)
                c
            member t.Gather() =
                let h_a = Array.zeroCreate<floatType> (int t.dArray.Size)
                t.dArray.CopyToHost(h_a)
                h_a

            member t.Gather'() =
                let h_a = Array2D.zeroCreate<floatType> num_rows num_cols
                t.dArray.CopyToHost(h_a)
                h_a

            interface IDisposable with
                member t.Dispose() = t.dArray.Dispose()

    module Numerics =
        type Layout = // cblas.h: typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
            | R = 101
            | C = 102
        type Transpose = // cblas.h: typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
            | NT = 111
            | T = 112
            | CT = 113

        let geam2 transa transb (alpha: floatType) (A:dMatrix) (beta: floatType) (B:dMatrix) (C:dMatrix) =
            let a_row = if transa = Operation.NonTranspose then A.num_rows else A.num_cols
            let a_col = if transa = Operation.NonTranspose then A.num_cols else A.num_rows
            let b_row = if transb = Operation.NonTranspose then B.num_rows else B.num_cols
            let b_col = if transb = Operation.NonTranspose then B.num_cols else B.num_rows
        
            if DoBoundsChecking then
                if a_row <> b_row then (failwithf "a_row <> b_row in geam2 (domatcopyT)! %i <> %i" a_row b_row)
                if a_col <> b_col then (failwithf "a_col <> b_col in geam2 (domatcopyT)! %i <> %i" a_col b_col)

            // For row major format, I invert the rows and columns.
            let lda = if transa = Operation.NonTranspose then a_col else a_row
            let ldb = if transa = Operation.NonTranspose then b_col else b_row
            let ldc = a_col

            // I also swap a_col and a_row in the call below.
            c.Geam(transa, transb, a_col, a_row, alpha, A.dArray, lda, B.dArray, ldb, beta, C.dArray, ldc)

        // B <- alpha * transpose(A)
        let omatcopyT(alpha, a:dMatrix, b:dMatrix) = geam2 Operation.Transpose Operation.Transpose alpha a 0.0f a b

        open ManagedCuda.NPP.NPPsExtensions
        let max (x: dMatrix) =
            use buf = new_dev<byte> (x.dArray.MaxGetBufferSize())
            use r = new_dev<floatType> 1
            x.dArray.Max(r,buf)
            r.[SizeT 0]

        let min (x: dMatrix) =
            use buf = new_dev<byte> (x.dArray.MaxGetBufferSize())
            use r = new_dev<floatType> 1
            x.dArray.Min(r,buf)
            r.[SizeT 0]

        let maxIndex (x: dMatrix) =
            use buf = new_dev<byte> (x.dArray.MaxIndxGetBufferSize())
            use r = new_dev<floatType> 1
            use r_idx = new_dev<int> 1
            x.dArray.MaxIndx(r,r_idx,buf)
            r_idx.[SizeT 0]

        let minIndex (x: dMatrix) =
            use buf = new_dev<byte> (x.dArray.MinIndxGetBufferSize())
            use r = new_dev<floatType> 1
            use r_idx = new_dev<int> 1
            x.dArray.MinIndx(r,r_idx,buf)
            r_idx.[SizeT 0]

        // Index of the absolute max. Despite what the function says in the description it still needs to be adjusted for C indexing.
        let amax (x:dMatrix) =
            let arg_incx = 1
            c.Max(x.dArray,arg_incx)-1

        // y <- alpha * x + y
        let axpy(alpha:floatType, x:dMatrix, y:dMatrix) =
            if DoBoundsChecking then
                if x.num_rows <> y.num_rows then failwith "x.num_rows <> y.num_rows"
            let arg_incx = 1
            let arg_incy = 1
            c.Axpy(alpha,x.dArray,arg_incx,y.dArray,arg_incy)

        let scal(alpha: floatType, x: dMatrix) =
            let arg_incx = 1
            let arg_incy = 1
            c.Scale(alpha,x.dArray,arg_incx)

        let dot(x: dMatrix, y: dMatrix) =
            if DoBoundsChecking then
                if x.num_rows <> y.num_rows then failwith "x.num_rows <> y.num_rows"
            let arg_incx = 1
            let arg_incy = 1
            c.Dot(x.dArray,arg_incx,y.dArray,arg_incy)

        // A <- A + alpha * x * yT (outer product)
        let ger(alpha: floatType, x:dMatrix, y:dMatrix, a:dMatrix) =
            if DoBoundsChecking then
                if x.num_cols <> 1 then failwith "x.num_cols <> 1"
                if y.num_cols <> 1 then failwith "y.num_cols <> 1"
                if x.num_rows <> a.num_rows then failwith "x.num_rows <> a.num_rows"
                if y.num_rows <> a.num_cols then failwith "y.num_rows <> a.num_cols"

            let arg_m = y.num_rows
            let arg_n = x.num_rows
            let arg_alpha = alpha
            let arg_incx = 1
            let arg_incy = 1
            let arg_lda = arg_m

            // As it happens, the wrapped c.Ger does not allow reversing the m and n arguments, so I have taken it out here.
            // This is an adjustment so it works for row major matrices.
            let _blasHandle = c.CublasHandle
            let _status = CudaBlasNativeMethods.cublasSger_v2(_blasHandle, arg_m, arg_n, ref alpha, x.dArray.DevicePointer, arg_incx, y.dArray.DevicePointer, arg_incy, a.dArray.DevicePointer, arg_lda)
            if (_status <> CublasStatus.Success) then raise (new CudaBlasException(_status))

        let asum(x:dMatrix) =
            let arg_incx = 1
            c.AbsoluteSum(x.dArray,arg_incx)

        let nrm2(x:dMatrix) =
            let arg_incx = 1
            c.Norm2(x.dArray,arg_incx)

        // O <- alpha * A * B + beta * O
        let gemm(alpha:floatType, a:dMatrix, b:dMatrix, beta:floatType, o:dMatrix) =
            if DoBoundsChecking then
                if a.num_cols <> b.num_rows then failwithf "a.num_cols=%i, b.num_rows=%i" a.num_cols b.num_rows
                if a.num_rows <> o.num_rows then failwithf "a.num_rows=%i, o.num_rows=%i" a.num_rows o.num_rows
                if b.num_cols <> o.num_cols then failwithf "b.num_cols=%i, o.num_cols=%i" a.num_cols o.num_cols

            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let m = a.num_rows
            let n = b.num_cols
            let k = b.num_rows
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
            c.Gemm(arg_transa,arg_transb,arg_m, arg_n, arg_k, arg_alpha, a.dArray, arg_lda, b.dArray, arg_ldb, arg_beta, o.dArray, arg_ldc)

        // y <- alpha * op(A) * x + beta * y
        let gemv(trans, alpha:floatType, a:dMatrix, x:dMatrix, beta:floatType, y:dMatrix) =
            let a_row = if trans = Operation.NonTranspose then a.num_rows else a.num_cols
            let a_col = if trans = Operation.NonTranspose then a.num_cols else a.num_rows

            if DoBoundsChecking then
                if a_col <> x.num_rows then failwithf "a.num_cols=%i, x.num_rows=%i" a.num_cols a.num_rows
                if a_row <> y.num_rows then failwithf "a.num_rows=%i, y.num_rows=%i" a.num_cols a.num_rows

            // I invert the transpose for row major's sake.
            let arg_trans = if trans = Operation.NonTranspose then Operation.Transpose else Operation.NonTranspose
            let arg_m = a.num_cols
            let arg_n = a.num_rows
            let arg_alpha = alpha
            let arg_lda = arg_m
            let arg_incx = 1
            let arg_beta = beta
            let arg_incy = 1
            c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, a.dArray, arg_lda, x.dArray, arg_incx, arg_beta, y.dArray, arg_incy)

        // The solver functions start here.

        let gesv(a:dMatrix, b:dMatrix) =
            let m = a.num_rows
            let n = a.num_cols

            if DoBoundsChecking then
                if m <> n then failwith "The matrix is not square"
                if m <> b.num_rows then failwith "The length of b does not equal the dimensions of a"

            use d_ipiv = new_dev<int> n

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n
    
            use d_a = new dMatrix(a.num_rows,a.num_cols)
            c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,a.dArray,n,a.dArray,n,0.0f,a.dArray,n) // Transpose using geam.

            let Lwork = s.GetrfBufferSize(m,n,a.dArray,arg_lda)
            use workspace = new_dev<floatType> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,a.dArray,arg_lda,workspace,d_ipiv,d_info)

            if DoBoundsChecking then
                let factorization_par = d_info.[SizeT 0]
                if factorization_par <> 0 then failwithf "Parameter %i in sgesv is incorrect." factorization_par
    
            s.Getrs(Operation.NonTranspose,arg_n,arg_nrhs,a.dArray,arg_lda,d_ipiv,b.dArray,arg_ldb,d_info)
    
            if d_info.[SizeT 0] = 0 then
                Some(b)
            else
                None

        let sysv(a:dMatrix, b:dMatrix) =
            let m = a.num_rows
            let n = a.num_cols

            if DoBoundsChecking then
                if m <> n then failwith "The matrix is not square"
                if m <> b.num_rows then failwith "The length of b does not equal the dimensions of a"

            let ipiv = Array.zeroCreate<floatType> n
            let work = Array.zeroCreate<floatType> 1

            use d_ipiv = new_dev<int> n

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n

            let Lwork = s.PotrfBufferSize(FillMode.Upper,n,a.dArray,arg_lda)
            use d_work = new_dev<floatType> Lwork
            use d_info = to_dev [|0|]

            s.Potrf(FillMode.Upper,arg_n,a.dArray,arg_lda,d_work,Lwork,d_info)

            if DoBoundsChecking then
                let factorization_par = d_info.[SizeT 0]
                if factorization_par <> 0 then failwithf "Parameter %i in ssysv is incorrect." factorization_par

            s.Potrs(FillMode.Upper,arg_n,arg_nrhs,a.dArray,arg_lda,b.dArray,arg_ldb,d_info)

            if d_info.[SizeT 0] = 0 then
                Some b
            else
                None

        let getrf(a:dMatrix) =
            let m = a.num_rows
            let n = a.num_cols

            if DoBoundsChecking then
                if m <> n then failwith "The matrix is not square"

            use d_ipiv = new_dev<int> (Microsoft.FSharp.Core.Operators.min m n)

            let arg_m = m
            let arg_n = n
            let arg_lda = m
    
            let Lwork = s.GetrfBufferSize(m,n,a.dArray,arg_lda)
            use workspace = new_dev<floatType> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,a.dArray,arg_lda,workspace,d_ipiv,d_info)

            if d_info.[SizeT 0] = 0 then
                Some(d_ipiv)
            else
                None

        let getri(a:dMatrix, d_ipiv:CudaDeviceVariable<int>) =
            let m = a.num_rows
            let n = a.num_cols

            if DoBoundsChecking then
                if m <> n then failwith "The matrix is not square"
                if m <> int d_ipiv.Size then failwith "The length of ipiv does not equal the dimensions of a"

            use d_ar_a = new_dev<CUdeviceptr> 1
            d_ar_a.[SizeT 0] <- a.dArray.DevicePointer

            use d_work = new_dev<floatType> (n * n)
            use d_ar_work = new_dev<CUdeviceptr> 1
            d_ar_work.[SizeT 0] <- d_work.DevicePointer

            let arg_n = n
            let arg_lda = n
            let arg_ldc = n
            let arg_lwork = n * n
            use d_info = to_dev [|0|]
            c.GetriBatchedS(arg_n,d_ar_a,arg_lda,d_ipiv,d_ar_work,arg_ldc,d_info,1)
            if d_info.[SizeT 0] = 0 then
                Some a
            else
                None

    let inline divup a b = (a+b-1)/b
    let numSm = ctx.GetDeviceInfo().MultiProcessorCount

    /// o <- f(x)
    type DeviceUnaryTransformModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" x)
                {
                    return "+op+"
                }
        
                // Device code
                __global__ void Map1Kernel(const "+FloatTypeCpp+"* A, "+FloatTypeCpp+"* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i]);
                        i += stride;
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map1Kernel")
        do  
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map1Kernel")

        member t.A(x: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            let o = new_dev<floatType> n
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
            o

        member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
            let n = int o.Size
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore

        member t.A(x: dMatrix) =
            let o = new dMatrix(x.num_rows,x.num_cols)
            t.A(x,o)
            o

        member t.A(x: dMatrix, o: dMatrix) =
            if x.rc <> o.rc then failwith "x.rc <> o.rc in DeviceUnaryTransformModule"
            t.A(x.dArray,o.dArray)

    /// o <- f(x,y)
    type DeviceBinaryTransformModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" x, "+FloatTypeCpp+" y)
                {
                    return "+op+"
                }
        
                // Device code
                __global__ void Map2Kernel(const "+FloatTypeCpp+"* A, const "+FloatTypeCpp+"* B, "+FloatTypeCpp+"* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i],B[i]);
                        i += stride;
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map2Kernel")
        do  
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map2Kernel")

        member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            let o = new_dev<floatType> n
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
            o

        member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
            let n = int o.Size
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore

        member t.A(x: dMatrix, y: dMatrix) =
            let o = new dMatrix(x.num_rows,x.num_cols)
            t.A(x,y,o)
            o

        member t.A(x: dMatrix, y: dMatrix, o: dMatrix) =
            if x.rc <> y.rc then failwith "x.rc <> o.rc in DeviceBinaryTransformModule"
            if y.rc <> o.rc then failwith "y.rc <> o.rc in DeviceBinaryTransformModule"
            t.A(x.dArray,y.dArray,o.dArray)

    /// o <- f(x,y,z)
    type DeviceTrinaryTransformModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" x, "+FloatTypeCpp+" y, "+FloatTypeCpp+" z)
                {
                    return "+op+"
                }
        
                // Device code
                __global__ void Map3Kernel(const "+FloatTypeCpp+"* A, const "+FloatTypeCpp+"* B, const "+FloatTypeCpp+"* C, "+FloatTypeCpp+"* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(A[i],B[i],C[i]);
                        i += stride;
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map3Kernel")
        do  
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map3Kernel")

        member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, z: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            let o = new_dev<floatType> n
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore
            o

        member t.A(x: CudaDeviceVariable<floatType>, y: CudaDeviceVariable<floatType>, z: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
            let n = int o.Size
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,z.DevicePointer,o.DevicePointer,n) |> ignore

        member t.A(x: dMatrix, y: dMatrix, z: dMatrix) =
            let o = new dMatrix(x.num_rows,x.num_cols)
            t.A(x,y,z,o)
            o

        member t.A(x: dMatrix, y: dMatrix, z: dMatrix, o: dMatrix) =
            if x.rc <> y.rc then failwith "x.rc <> o.rc in DeviceTrinaryTransformModule"
            if y.rc <> z.rc then failwith "y.rc <> z.rc in DeviceTrinaryTransformModule"
            if z.rc <> o.rc then failwith "z.rc <> o.rc in DeviceTrinaryTransformModule"
            t.A(x.dArray,y.dArray,z.dArray,o.dArray)

    /// o <- sum(f(x))
    type DeviceUnaryMapSumModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" x)
                {
                    return "+op+"
                }
        
                __device__ inline "+FloatTypeCpp+" warpDownReduce("+FloatTypeCpp+" value){
	                for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	                return value;
                }

                // Device code
                __global__ void MapSumKernel(const "+FloatTypeCpp+"* A, "+FloatTypeCpp+"* O, const int N)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                __shared__ "+FloatTypeCpp+" temp[32];
                    if (threadIdx.x < 32) temp[threadIdx.x] = 0.0f; "+FloatTypeCpp+" acc = 0.0f;
	                while (i < N)
	                {
		                acc += op(A[i]);
		                i += stride;
	                }
	                __syncthreads(); "+FloatTypeCpp+" out_partial = warpDownReduce(acc);
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	                __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	                if (threadIdx.x == 0) atomicAdd(O, out_partial);
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapSumKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapSumKernel")

        member t.A(x: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            use o = new_dev<floatType> 1
            o.Memset(0u)
            let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer,n) |> ignore
            o.[SizeT 0]

        member t.A(x: dMatrix) =
            t.A(x.dArray)

    /// o <- sum(f(x,y))
    type DeviceBinaryMapSumModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" x, "+FloatTypeCpp+" y)
                {
                    return "+op+"
                }
        
                __device__ inline "+FloatTypeCpp+" warpDownReduce("+FloatTypeCpp+" value){
	                for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	                return value;
                }

                // Device code
                __global__ void Map2SumKernel(const "+FloatTypeCpp+"* A, const "+FloatTypeCpp+"* B, "+FloatTypeCpp+"* O, const int N)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                __shared__ "+FloatTypeCpp+" temp[32];
                    if (threadIdx.x < 32) temp[threadIdx.x] = 0.0f; "+FloatTypeCpp+" acc = 0.0f;
	                while (i < N)
	                {
		                acc += op(A[i],B[i]);
		                i += stride;
	                }
	                __syncthreads(); "+FloatTypeCpp+" out_partial = warpDownReduce(acc);
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	                __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	                if (threadIdx.x == 0) atomicAdd(O, out_partial);
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"Map2SumKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"Map2SumKernel")

        member t.A(x: CudaDeviceVariable<floatType>,y: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            use o = new_dev<floatType> 1
            o.Memset(0u)
            let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,y.DevicePointer,o.DevicePointer,n) |> ignore
            o.[SizeT 0]

        member t.A(x: dMatrix,y: dMatrix) =
            if x.rc <> y.rc then failwith "x.rc <> y.rc in DeviceBinaryMapSumModule"
            t.A(x.dArray,y.dArray)

    /// o <- f(coef_x,x)
    type DeviceUnaryCoefTransformModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" coef_x, "+FloatTypeCpp+" x)
                {
                    return "+op+"
                }
        
                // Device code
                __global__ void MapCoefKernel(const "+FloatTypeCpp+" coef_A, const "+FloatTypeCpp+"* A, "+FloatTypeCpp+"* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(coef_A,A[i]);
                        i += stride;
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapCoefKernel")
        do  
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapCoefKernel")

        member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            let o = new_dev<floatType> n
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore
            o

        member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
            let n = int o.Size
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,o.DevicePointer,n) |> ignore

        member t.A(coef_x, x: dMatrix) =
            let o = new dMatrix(x.num_rows,x.num_cols)
            t.A(coef_x,x,o)
            o

        member t.A(coef_x, x: dMatrix, o: dMatrix) =
            if x.rc <> o.rc then failwith "y.rc <> o.rc in DeviceUnaryCoefTransformModule"
            t.A(coef_x,x.dArray,o.dArray)

    /// o <- f(coef_x,x,coef_y,y)
    type DeviceBinaryCoefTransformModule(op: string) = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" op("+FloatTypeCpp+" coef_x, "+FloatTypeCpp+" x, "+FloatTypeCpp+" coef_y, "+FloatTypeCpp+" y)
                {
                    return "+op+"
                }
        
                // Device code
                __global__ void MapCoef2Kernel(const "+FloatTypeCpp+" coef_A, const "+FloatTypeCpp+"* A, const "+FloatTypeCpp+" coef_B, const "+FloatTypeCpp+"* B, "+FloatTypeCpp+"* O, const int N)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    while (i < N)
                    {
                        O[i] = op(coef_A,A[i],coef_B,B[i]);
                        i += stride;
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"MapCoef2Kernel")
        do  
            try k.Compile([||])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"MapCoef2Kernel")

        member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>,coef_y: floatType, y: CudaDeviceVariable<floatType>) =
            let n = int x.Size
            let o = new_dev<floatType> n
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y, y.DevicePointer,o.DevicePointer,n) |> ignore
            o

        member t.A(coef_x: floatType, x: CudaDeviceVariable<floatType>, coef_y: floatType, y: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>) =
            let n = int o.Size
            let gridSize = min (divup n block_size) 16*numSm
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, coef_x,x.DevicePointer,coef_y,y.DevicePointer,o.DevicePointer,n) |> ignore

        member t.A(coef_x, x: dMatrix, coef_y, y: dMatrix) =
            let o = new dMatrix(x.num_rows,x.num_cols)
            t.A(coef_x,x,coef_y,y,o)
            o

        member t.A(coef_x, x: dMatrix, coef_y, y: dMatrix, o: dMatrix) =
            if x.rc <> y.rc then failwith "y.rc <> y.rc in DeviceBinaryCoefTransformModule"
            if y.rc <> o.rc then failwith "y.rc <> o.rc in DeviceBinaryCoefTransformModule"
            t.A(coef_x,x.dArray,coef_y,y.dArray,o.dArray)

    /// Expands a vector into a matrix by repeatedly copying its elements.
    /// http://stackoverflow.com/questions/25452519/how-to-efficiently-repeat-a-vector-to-a-matrix-in-cuda
    type DeviceTileModule() = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __global__ void expand_kernel(const "+FloatTypeCpp+"* vector, const unsigned vlen, "+FloatTypeCpp+"* matrix, const unsigned mdim, const unsigned col_major){
                    if (col_major){
                    int idx = threadIdx.x+blockIdx.x*mdim; "+FloatTypeCpp+" myval = vector[blockIdx.x];
                    while (idx < ((blockIdx.x+1)*mdim)){
                        matrix[idx] = myval;
                        idx += blockDim.x;
                        }
                    }
                    else{
                    int idx = threadIdx.x + blockDim.x * blockIdx.x; "+FloatTypeCpp+" myval = vector[idx%vlen];
                    while (idx < mdim*vlen){
                        matrix[idx] = myval;
                        idx += gridDim.x*blockDim.x;
                        }
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"expand_kernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"expand_kernel")

        /// [1,2,3] -> [1,1,2,2,3,3]
        /// Expands the vector along the rows.
        member t.AR(num_repeat: uint32, x: dMatrix) =
            if x.num_cols <> 1 then failwith "The tile kernel must take a vector. x.num_cols <> 1!"
            let order = 1u
            let y = new dMatrix(x.num_rows, int num_repeat)
            let n = uint32 x.dArray.Size
            let gridSize = n
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,n,y.dArray.DevicePointer,num_repeat,order) |> ignore
            y

        /// [1,2,3] -> [1,2,3,1,2,3]
        /// Expands the vector along the columns.
        member t.AC(num_repeat: uint32, x: dMatrix) =
            if x.num_cols <> 1 then failwith "The tile kernel must take a vector. x.num_cols <> 1!"
            let order = 0u
            let y = new dMatrix(x.num_rows, int num_repeat)
            let n = uint32 x.dArray.Size
            let gridSize = min (2*numSm*(1024/block_size)) (divup (n*num_repeat |> int) block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,n,y.dArray.DevicePointer,num_repeat,order) |> ignore
            y

    ///o <- diag(x)
    type DeviceDiagonalModule() = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                // Device code
                __global__ void DiagonalKernel(const "+FloatTypeCpp+"* A, "+FloatTypeCpp+"* O, const int N, const int M)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                while (i < N)
	                {
		                O[i] = A[i+i*M];
		                i += stride;
	                }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"DiagonalKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"DiagonalKernel")

        /// Note that this is for row major. For col major A[i+i*M] would have to be A[i+i*N]
        member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>, n, m) =
            let gridSize = min (2*numSm*(1024/block_size)) (divup (n*m) block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer,o.DevicePointer, min m n, m) |> ignore

        member t.A(x: dMatrix) =
            let l = min x.num_rows x.num_cols
            let o = new dMatrix(l,1)
            t.A(x.dArray, o.dArray, x.num_rows, x.num_cols)
            o

    /// o <- det(x_qr_factorized)
    /// Needs to be qr_factorized first.
    type DeviceDetModule() = 
        let grid_size = 32
        let block_size = 256

        let workspace = new_dev<floatType> grid_size

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __device__ inline "+FloatTypeCpp+" warpDownReduce("+FloatTypeCpp+" value){
	                for (int i = 16; i>0; i = i / 2) value *= __shfl_down(value, i);
	                return value;
                }

                __device__ inline "+FloatTypeCpp+" blockReduce("+FloatTypeCpp+" value){
	                __shared__ "+FloatTypeCpp+" temp[32];
                    if (threadIdx.x < 32) temp[threadIdx.x] = 1.0f; "+FloatTypeCpp+" out_partial = warpDownReduce(value);
                    __syncthreads();
	                if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
                    __syncthreads();
	                if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
                    return out_partial;
                }

                // Device code
                __global__ void DeterminantKernel(const "+FloatTypeCpp+"* A, const int* ipiv, "+FloatTypeCpp+"* O, const int N, const int M)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
                    if (threadIdx.x < 32) O[threadIdx.x] = 1.0f;
	                const int stride = blockDim.x * gridDim.x; "+FloatTypeCpp+" acc = 1.0f;
	                while (i < N)
	                {
		                if (ipiv[i] != i+1) acc *= -A[i+i*M];
                        else acc *= A[i+i*M];
		                i += stride;
	                }   
                        "+FloatTypeCpp+" out = blockReduce(acc);
                        if (threadIdx.x == 0) {
                            O[blockIdx.x] = out;
                            }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"DeterminantKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"DeterminantKernel")

        member t.A(x: CudaDeviceVariable<floatType>, ipiv: CudaDeviceVariable<int>, n, m) =
            let gridSize = min grid_size (divup (n*m) block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.DevicePointer, ipiv.DevicePointer, workspace.DevicePointer, n, m) |> ignore
            workspace.Gather() |> Array.reduce ((*)) // Cuda does not allow inter block synchronization inside the kernel, so I have the CPU make the final reduce here.

        member t.A(x: dMatrix, ipiv: CudaDeviceVariable<int>) =
            if x.num_rows <> x.num_cols then failwith "x.num_rows <> x.num_cols"
            if x.num_cols <> int ipiv.Size then failwith "x.num_cols <> int ipiv.Size"
            t.A(x.dArray, ipiv, x.num_rows, x.num_cols)

    ///diag(o) += x
    type DeviceDiagonalAddModule() = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                // Device code
                __global__ void DiagonalAddKernel(const "+FloatTypeCpp+"* A, "+FloatTypeCpp+"* O, const int N, const int M)
                {
	                int i = blockDim.x * blockIdx.x + threadIdx.x;
	                const int stride = blockDim.x * gridDim.x;
	                while (i < N)
	                {
		                O[i+i*M] += A[i];
		                i += stride;
	                }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"DiagonalAddKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"DiagonalAddKernel")

        /// Note that this is for row major. For col major O[i+i*N] would have to be O[i+i*N]
        /// Is mutable.
        member t.A(x: CudaDeviceVariable<floatType>, o: CudaDeviceVariable<floatType>, n, m) =
            let gridSize = min (2*numSm*(1024/block_size)) (divup (n*m) block_size)
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, o.DevicePointer,x.DevicePointer, min m n, m) |> ignore

        member t.A(x: dMatrix, o: dMatrix) =
            if int o.dArray.Size <> min x.num_rows x.num_cols then failwith "o.dArray.Size <> min x.num_rows x.num_cols"
            t.A(x.dArray, o.dArray, x.num_rows, x.num_cols)

    type DeviceGetSliceModule() = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __global__ void getSliceKernel(const "+FloatTypeCpp+"* matrix, "+FloatTypeCpp+"* out_matrix, const int start_row, const int end_row, const int num_rows, const int start_col, const int end_col, const int num_cols, const unsigned col_major){
                    const int stride = blockDim.x * gridDim.x;
                    if (col_major){
                        int i = threadIdx.x+blockIdx.x*blockDim.x;
                        const int row_stride = end_row-start_row+1;
                        const int col_stride = end_col-start_col+1;
                        while (1) {
                            const int row_i = i % row_stride;
                            const int col_i = i / row_stride;
                            const int row = start_row+row_i;
                            const int col = start_col+col_i;
                            const int idx = row+col*num_rows;
                            if (row_i < row_stride && col_i < col_stride) {
                                out_matrix[i] = matrix[idx];
                                i += stride;
                            } else return;
                        }
                    }
                    else{
                        int i = threadIdx.x+blockIdx.x*blockDim.x;
                        const int row_stride = end_row-start_row+1;
                        const int col_stride = end_col-start_col+1;
                        while (1) {
                            const int row_i = i / col_stride;
                            const int col_i = i % col_stride;
                            const int row = start_row+row_i;
                            const int col = start_col+col_i;
                            const int idx = col+row*num_cols;
                            if (row_i < row_stride && col_i < col_stride) {
                                out_matrix[i] = matrix[idx];
                                i += stride;
                            } else return;
                        }
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"getSliceKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"getSliceKernel")

        /// For matrices stored in row major order.
        /// Zero based indexing.
        member t.AR(x: dMatrix, start_row, end_row, start_col, end_col) =
            if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
            if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
            let order = 0u
            let row_stride = end_row-start_row+1
            let col_stride = end_col-start_col+1
            let y = new dMatrix(row_stride, col_stride)
            let n = row_stride*col_stride
            let gridSize = divup n block_size
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore
            y

        /// For matrices stored in column major order.
        /// Zero based indexing.
        member t.AC(x: dMatrix, start_row, end_row, start_col, end_col) =
            if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
            if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
            let order = 1u
            let row_stride = end_row-start_row+1
            let col_stride = end_col-start_col+1
            let y = new dMatrix(row_stride, col_stride)
            let n = row_stride*col_stride
            let gridSize = divup n block_size
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore
            y

    type DeviceSetSliceModule() = 
        let block_size = 256

        let kernel_code = "
            //Kernel code:
            extern \"C\" {
                __global__ void setSliceKernel("+FloatTypeCpp+"* matrix, const "+FloatTypeCpp+"* out_matrix, const int start_row, const int end_row, const int num_rows, const int start_col, const int end_col, const int num_cols, const unsigned col_major){
                    const int stride = blockDim.x * gridDim.x;
                    if (col_major){
                        int i = threadIdx.x+blockIdx.x*blockDim.x;
                        const int row_stride = end_row-start_row+1;
                        const int col_stride = end_col-start_col+1;
                        while (1) {
                            const int row_i = i % row_stride;
                            const int col_i = i / row_stride;
                            const int row = start_row+row_i;
                            const int col = start_col+col_i;
                            const int idx = row+col*num_rows;
                            if (row_i < row_stride && col_i < col_stride) {
                                matrix[idx] = out_matrix[i];
                                i += stride;
                            } else return;
                        }
                    }
                    else{
                        int i = threadIdx.x+blockIdx.x*blockDim.x;
                        const int row_stride = end_row-start_row+1;
                        const int col_stride = end_col-start_col+1;
                        while (1) {
                            const int row_i = i / col_stride;
                            const int col_i = i % col_stride;
                            const int row = start_row+row_i;
                            const int col = start_col+col_i;
                            const int idx = col+row*num_cols;
                            if (row_i < row_stride && col_i < col_stride) {
                                matrix[idx] = out_matrix[i];
                                i += stride;
                            } else return;
                        }
                    }
                }
            }

            "
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,"setSliceKernel")
        do  
            try k.Compile([|"-arch=compute_30"|])
            with 
            | :? NVRTCException as x -> 
                printfn "%s" (k.GetLogAsString())
                reraise()

        let kernel = ctx.LoadKernelPTX(k.GetPTX(),"setSliceKernel")

        /// For matrices stored in row major order.
        /// Zero based indexing. Is mutable.
        member t.AR(x: dMatrix, y: dMatrix, start_row, end_row, start_col, end_col) =
            if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
            if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
            let order = 0u
            let row_stride = end_row-start_row+1
            let col_stride = end_col-start_col+1
            let n = row_stride*col_stride
            let gridSize = divup n block_size
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore

        /// For matrices stored in column major order.
        /// Zero based indexing. Is mutable.
        member t.AC(x: dMatrix, y: dMatrix, start_row, end_row, start_col, end_col) =
            if (start_row < 0 || start_col < 0) then failwith "start_row < 0 || start_col < 0"
            if (end_row >= x.num_rows || start_col >= x.num_cols) then failwith "end_row >= x.num_rows || start_col >= x.num_cols"
            let order = 1u
            let row_stride = end_row-start_row+1
            let col_stride = end_col-start_col+1
            let n = row_stride*col_stride
            let gridSize = divup n block_size
            kernel.GridDimensions <- dim3(gridSize)
            kernel.BlockDimensions <- dim3(block_size)
            kernel.RunAsync(str.Stream, x.dArray.DevicePointer,y.dArray.DevicePointer,start_row, end_row, x.num_rows, start_col, end_col, x.num_cols, order) |> ignore


    // Might be good to make these lazy or to precompile them at some point.
    let scalarMatrixAddModule = new DeviceBinaryCoefTransformModule "coef_x + coef_y*x;"
    let sumModule = new DeviceUnaryMapSumModule "x;"
    let tileModule = DeviceTileModule()
    let hadamaradMultiplicationModule = new DeviceBinaryTransformModule "x*y;"
    let hadamaradDivisionModule = new DeviceBinaryTransformModule "x/y;"
    let diagonalModule = new DeviceDiagonalModule()
    let addDiagModule = new DeviceDiagonalAddModule()
    let determinantModule = new DeviceDetModule()
    let getsliceModule = new DeviceGetSliceModule()
    /// Is mutable.
    let setsliceModule = new DeviceSetSliceModule()

    let powModule = new DeviceBinaryTransformModule "powf(x,y);"
    ///powf(x,coef_x)
    let scalarpowModule = new DeviceUnaryCoefTransformModule "powf(x,coef_x);"
    ///powf(coef_x,x)
    let scalarpow2Module = new DeviceUnaryCoefTransformModule "powf(coef_x,x);"
    let atan2Module = new DeviceBinaryTransformModule "atan2f(x,y);"
    let scalardivModule = new DeviceUnaryCoefTransformModule "coef_x/x;"
    ///atan2f(x,coef_x)
    let atan2scalarModule = new DeviceUnaryCoefTransformModule "atan2f(x,coef_x);"
    ///atan2f(coef_x,x)
    let atan3scalarModule = new DeviceUnaryCoefTransformModule "atan2f(coef_x,x);"

    let logModule = new DeviceUnaryTransformModule "logf(x);"
    let log10Module = new DeviceUnaryTransformModule "log10f(x);"
    let expModule = new DeviceUnaryTransformModule "expf(x);"
    let sinModule = new DeviceUnaryTransformModule "sinf(x);"
    let cosModule = new DeviceUnaryTransformModule "cosf(x);"
    let tanModule = new DeviceUnaryTransformModule "tanf(x);"
    let sqrtModule = new DeviceUnaryTransformModule "sqrtf(x);"
    let sinhModule = new DeviceUnaryTransformModule "sinhf(x);"
    let coshModule = new DeviceUnaryTransformModule "coshf(x);"
    let tanhModule = new DeviceUnaryTransformModule "tanhf(x);"
    let asinModule = new DeviceUnaryTransformModule "asinf(x);"
    let acosModule = new DeviceUnaryTransformModule "acosf(x);"
    let atanModule = new DeviceUnaryTransformModule "atanf(x);"
    let absModule = new DeviceUnaryTransformModule "fabsf(x);"
    let signModule = new DeviceUnaryCoefTransformModule "copysignf(coef_x,x);"
    let floorModule = new DeviceUnaryTransformModule "floorf(x);"
    let ceilModule = new DeviceUnaryTransformModule "ceilf(x);"
    let roundModule = new DeviceUnaryTransformModule "roundf(x);"
    let reluModule = DeviceUnaryTransformModule "x > 0.0f ? x : 0.0f;"
    let sigmoidModule = DeviceUnaryTransformModule "1.0f / (1.0f + expf(-x));"

    type dMatrix with
        member t.Item
            with get(a: int, b: int) = t.dArray.[b+a*t.num_cols |> SizeT]
            and set(a: int, b: int) (value: floatType) = t.dArray.[b+a*t.num_cols |> SizeT] <- value

        member t.GetSlice(rowStart: int option, rowFinish : int option,
                             colStart: int option, colFinish : int option) =
            let rowStart = defaultArg rowStart 0
            let rowFinish = defaultArg rowFinish (t.num_rows-1)
            let colStart = defaultArg colStart 0
            let colFinish = defaultArg colFinish (t.num_cols-1)
            getsliceModule.AR(t,rowStart,rowFinish,colStart,colFinish)

        member t.GetSlice(row: int, colStart: int option, colFinish: int option) =
            let colStart = defaultArg colStart 0
            let colFinish = defaultArg colFinish t.num_cols-1
            getsliceModule.AR(t,row,row,colStart,colFinish)

        member t.GetSlice(rowStart: int option, rowFinish: int option, col: int) =
            let rowStart = defaultArg rowStart 0
            let rowFinish = defaultArg rowFinish t.num_rows-1
            getsliceModule.AR(t,rowStart,rowFinish,col,col)

    type cudaBackend() =
            // Numerics
            member t.Add_V_V(x: dMatrix, y: dMatrix) =
                    let y' = y.copy()
                    Numerics.axpy(1.f, x, y')
                    y'
            // Numerics
            member t.Add_S_V(x: floatType, y: dMatrix) =
                    let x' = y.zeroLike()
                    Numerics.axpy(1.f, y, x')
                    x'
            // Numerics
            member t.Mul_S_V(alpha: floatType, x: dMatrix) =
                    let x' = x.copy()
                    Numerics.scal(alpha, x')
                    x'
            // Numerics
            member t.Sub_V_V(x: dMatrix, y: dMatrix) =
                    let x' = x.copy()
                    Numerics.axpy(-1.f, y, x')
                    x'
            // Numerics
            member t.Mul_Dot_V_V(x: dMatrix, y: dMatrix) =
                    Numerics.dot(x, y)
            // Numerics
            member t.Mul_Out_V_V(x: dMatrix, y: dMatrix) =
                    let z = new dMatrix(x.num_rows, y.num_rows)
                    Numerics.ger(1.f, x, y, z)
                    z

            // Non-Numerics
            member t.Sub_S_V(alpha: floatType, x: dMatrix) =
                    scalarMatrixAddModule.A(alpha,x,-1.0f,x) // The second x does nothing here. alpha = coef_x, -1.0f = coef_y
            // Non-Numerics
            member t.Sub_V_S(x: dMatrix, alpha: floatType) =
                    scalarMatrixAddModule.A(-alpha,x,1.0f,x) // The second x does nothing here. alpha = coef_x, 1.0f = coef_y
            // Non-Numerics
            member t.Sub_S_M(alpha: floatType, x: dMatrix) =
                    scalarMatrixAddModule.A(alpha,x,-1.0f,x) // The second x does nothing here. alpha = coef_x, -1.0f = coef_y
            // Non-Numerics
            member t.Sub_M_S(x: dMatrix, alpha: floatType) =
                    scalarMatrixAddModule.A(-alpha,x,1.0f,x) // The second x does nothing here. alpha = coef_x, 1.0f = coef_y
            // Numerics
            member t.L1Norm_V(x) =
                    Numerics.asum(x)
            // Numerics
            member t.L2Norm_V(x) =
                    Numerics.nrm2(x)
            // Numerics
            member t.SupNorm_V(x) =
                    let i = Numerics.amax(x)
                    abs x.dArray.[i - SizeT 1]
            // Non-Numerics
            member t.Sum_V(x: dMatrix) =
                    sumModule.A(x)
            // Numerics
            member t.Add_M_M(x: dMatrix, y: dMatrix) =
                    let y' = y.copy()
                    Numerics.axpy(1.f, x, y')
                    y'
            // Numerics
            member t.Add_S_M(alpha: floatType, x: dMatrix) =
                    scalarMatrixAddModule.A(alpha,x,1.0f,x) // The second x does nothing here. alpha = coef_x, -1.0f = coef_y
            // Numerics
            member t.Add_V_MCols(x: dMatrix, y: dMatrix) =
                    let x' = t.RepeatReshapeCopy_V_MCols(uint32 y.num_cols, x)
                    Numerics.axpy(1.f, y, x')
                    x'
            // Numerics
            member t.Mul_S_M(alpha: floatType, x: dMatrix) =
                    let x' = x.copy()
                    Numerics.scal(alpha, x')
                    x'
            // Numerics
            member t.Sub_M_M(x: dMatrix, y: dMatrix) =
                    let x' = x.copy()
                    Numerics.axpy(-1.f, y, x')
                    x'
            // Numerics
            member t.Mul_M_M(x: dMatrix, y: dMatrix) =
                    let z = new dMatrix(x.num_rows, y.num_cols)
                    Numerics.gemm(1.f, x, y, 0.f, z)
                    z
            // Numerics
            member t.Mul_M_M_Add_V_MCols(x, y: dMatrix, z) =
                    let n = uint32 y.num_cols
                    let z' = t.RepeatReshapeCopy_V_MCols(n, z)
                    Numerics.gemm(1.f, x, y, 1.f, z')
                    z'

            // Non-Numerics
            member t.Mul_Had_V_V(x:dMatrix, y) =
                    hadamaradMultiplicationModule.A(x,y)
            // Non-Numerics
            member t.Mul_Had_M_M(x:dMatrix, y) =
                    hadamaradMultiplicationModule.A(x,y)
            // Non-Numerics
            member t.Div_Had_V_V(x:dMatrix, y) =
                    hadamaradDivisionModule.A(x,y)
            // Non-Numerics
            member t.Div_Had_M_M(x:dMatrix, y) =
                    hadamaradDivisionModule.A(x,y)
            // Numerics
            member t.Mul_M_V(x: dMatrix, y: dMatrix) =
                    let z = new dMatrix(x.num_rows,1)
                    Numerics.gemv(Operation.NonTranspose, 1.f, x, y, 0.f, z)
                    z
            // Numerics
            member t.Mul_M_V_Add_V(x, y, z: dMatrix) =
                    let z' = z.copy()
                    Numerics.gemv(Operation.NonTranspose, 1.f, x, y, 1.f, z')
                    z'
            // Numerics
            member t.Mul_V_M(x: dMatrix, y: dMatrix) =
                    let z = new dMatrix(x.num_cols, 1)
                    Numerics.gemv(Operation.Transpose, 1.f, y, x, 0.f, z)
                    z
            // Numerics extension
            member t.Transpose_M(x: dMatrix) =
                    let x' = new dMatrix(x.num_cols,x.num_rows)
                    Numerics.omatcopyT(1.f, x, x')
                    x'
            // Non-Numerics
            member t.Sum_M(x: dMatrix) =
                    sumModule.A(x)
            // Numerics
            member t.Solve_M_V(x, y) =
                    Numerics.gesv(x, y)
            // Numerics
            member t.SolveSymmetric_M_V(x, y) =
                    Numerics.sysv(x, y)
            // Non-Numerics
            member t.Diagonal_M(x) =
                    diagonalModule.A(x)
            // Numerics
            member t.Inverse_M(x: dMatrix) =
                    let x' = x.copy()
                    let ipiv = Numerics.getrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = Numerics.getri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            // Numerics
            member t.Det_M(x: dMatrix) =
                    let x' = x.copy()
                    let ipiv = Numerics.getrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let det = determinantModule.A(x',ipiv)
                        Some(det)
                    | _ -> None

            // Non-Numerics
            member t.ReshapeCopy_V_MRows(m: int, a: dMatrix) =
                    let a' = a.copy()
                    new dMatrix(a.num_rows/m,m,a'.dArray)
            // Non-Numerics
            member t.RepeatReshapeCopy_V_MRows(m, x) =
                    tileModule.AC(m,x)
            // Non-Numerics
            member t.RepeatReshapeCopy_V_MCols(n, x: dMatrix) =
                    tileModule.AR(n,x)
            // Non-Numerics
            member t.RepeatReshapeCopy_V_MRows(m:int, x) =
                    tileModule.AC(uint32 m,x)
            // Non-Numerics
            member t.RepeatReshapeCopy_V_MCols(n:int, x: dMatrix) =
                    tileModule.AR(uint32 n,x)

            /// x_vector ** y_vector
            member t.Pow(x: dMatrix,y: dMatrix) = powModule.A(x,y)
            /// x_vector ** y_scalar
            member t.Pow2(x: dMatrix,y: floatType) = scalarpowModule.A(y,x)
            /// x_scalar ** y_vector
            member t.Pow3(x: floatType,y: dMatrix) = scalarpow2Module.A(x,y)
            member t.ATan2(x: dMatrix,y) = atan2Module.A(x,y)
            member t.ScalarDiv(x:floatType,y:dMatrix) = scalardivModule.A(x,y)
            ///(x: dMatrix, y: floatType)
            member t.ATan2Scalar(x: dMatrix, y: floatType) = atan2scalarModule.A(y,x)
            ///(x: floatType, y: dMatrix)
            member t.ATan2Scalar'(x: floatType, y: dMatrix) = atan3scalarModule.A(x,y)

            member t.Log(x:dMatrix) = logModule.A(x)
            member t.Log10(x:dMatrix) = log10Module.A(x)
            member t.Exp(x:dMatrix) = expModule.A(x)
            member t.Sin(x:dMatrix) = sinModule.A(x)
            member t.Cos(x:dMatrix) = cosModule.A(x)
            member t.Tan(x:dMatrix) = tanModule.A(x)
            member t.Sqrt(x:dMatrix) = sqrtModule.A(x)
            member t.Sinh(x:dMatrix) = sinhModule.A(x)
            member t.Cosh(x:dMatrix) = coshModule.A(x)
            member t.Tanh(x:dMatrix) = tanhModule.A(x)
            member t.Asin(x:dMatrix) = asinModule.A(x)
            member t.Acos(x:dMatrix) = acosModule.A(x)
            member t.Atan(x:dMatrix) = atanModule.A(x)
            member t.Abs(x:dMatrix) = absModule.A(x)
            member t.Sign(x:dMatrix) = signModule.A(1.0f,x)
            member t.Floor(x:dMatrix) = floorModule.A(x)
            member t.Ceil(x:dMatrix) = ceilModule.A(x)
            member t.Round(x:dMatrix) = roundModule.A(x)
            
            /// Is mutable.
            member t.SetSliceMut(x,y,start_row,end_row,start_col,end_col) = setsliceModule.AR(x,y,start_row,end_row,start_col,end_col)

            member t.Relu(x: dMatrix) = reluModule.A(x)
            member t.Sigmoid(x: dMatrix) = sigmoidModule.A(x)

            /// Is mutable.
            member t.AddDiagonalMut(x:dMatrix,y) = addDiagModule.A(x,y)

            member t.ReshapeCopy_MRows_V(x:dMatrix) = new dMatrix(x.num_rows*x.num_cols, 1, (x.copy().dArray))

            member t.Max(x: dMatrix) = Numerics.max x
            member t.MaxIdx(x: dMatrix) = Numerics.maxIndex x
            member t.Min(x: dMatrix) = Numerics.min x
            member t.MinIdx(x: dMatrix) = Numerics.minIndex x