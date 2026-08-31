const _CHECKPOINT_SCHEMA = 2
const _CHECKPOINT_MAGIC = UInt8[0x4d, 0x4f, 0x52, 0x46, 0x45, 0x43, 0x50, 0x32]

mutable struct CheckpointSession
    options::CheckpointOptions
    fingerprint::String
    manifest::Dict{String, Any}
end

"""
    checkpoint_fingerprint_data(callable)

Return immutable, deterministic data identifying a nonlinear callable used by a
checkpointed model. Extend this method for application-defined kernels. The default
returns `nothing`, causing checkpoint creation to reject the opaque callable rather
than treating a caller label or a source location as mathematical identity.
"""
checkpoint_fingerprint_data(::Any) = nothing

function _digest_text!(ctx, value)
    SHA.update!(ctx, codeunits(string(value)))
    SHA.update!(ctx, UInt8[0])
    return nothing
end

function _fingerprint_value!(ctx, value)
    _digest_text!(ctx, typeof(value))
    if value === nothing || value isa Number || value isa Symbol ||
       value isa AbstractString ||
       value isa Bool
        _digest_text!(ctx, value)
    elseif value isa SparseMatrixCSC
        _fingerprint_value!(ctx, size(value))
        _fingerprint_value!(ctx, value.colptr)
        _fingerprint_value!(ctx, value.rowval)
        _fingerprint_value!(ctx, value.nzval)
    elseif value isa AbstractArray
        _fingerprint_value!(ctx, size(value))
        if isbitstype(eltype(value))
            contiguous = value isa Array ? value : Array(value)
            bytes = reinterpret(UInt8, vec(contiguous))
            SHA.update!(ctx, bytes)
        else
            foreach(item -> _fingerprint_value!(ctx, item), value)
        end
    elseif value isa Tuple || value isa NamedTuple
        foreach(item -> _fingerprint_value!(ctx, item), value)
    elseif value isa Function
        data = checkpoint_fingerprint_data(value)
        data === nothing && throw(ArgumentError(
            "checkpoint cannot safely fingerprint opaque callable $(typeof(value)); " *
            "define MORFE.checkpoint_fingerprint_data(::$(typeof(value))) to return " *
            "deterministic mathematical identity data"))
        _fingerprint_value!(ctx, data)
    elseif value isa AbstractDict
        ordered = sort!(collect(keys(value)); by = repr)
        for key in ordered
            _fingerprint_value!(ctx, key)
            _fingerprint_value!(ctx, value[key])
        end
    else
        isstructtype(typeof(value)) || throw(ArgumentError(
            "checkpoint cannot fingerprint value of type $(typeof(value))"))
        for field in 1:fieldcount(typeof(value))
            _fingerprint_value!(ctx, getfield(value, field))
        end
    end
    return nothing
end

function _problem_fingerprint(model, spectral, resonance_set, mset, conj_perm,
        options::ParametrisationOptions)
    ctx = SHA.SHA256_CTX()
    _fingerprint_value!(ctx, _CHECKPOINT_SCHEMA)
    _fingerprint_value!(ctx, model)
    _fingerprint_value!(ctx, spectral)
    _fingerprint_value!(ctx, resonance_set)
    _fingerprint_value!(ctx, mset.exponents)
    _fingerprint_value!(ctx, conj_perm)
    _fingerprint_value!(ctx,
        (options.backend, options.grouping,
            options.residual_check, options.residual_tolerance,
            options.max_refinement_steps))
    return bytes2hex(SHA.digest!(ctx))
end

_manifest_path(root) = joinpath(root, "manifest.toml")
_lock_path(root) = joinpath(root, ".writer-lock")

function _atomic_manifest(path, manifest)
    temporary = path * ".tmp.$(getpid())"
    open(temporary, "w") do io
        TOML.print(io, manifest; sorted = true)
        flush(io)
    end
    mv(temporary, path; force = true)
    return nothing
end

function _with_checkpoint_lock(f, root)
    lock = _lock_path(root)
    try
        mkdir(lock)
    catch err
        isdir(lock) && throw(ArgumentError(
            "checkpoint $root is locked by another writer; remove $lock only after verifying no writer is active"))
        rethrow(err)
    end
    try
        return f()
    finally
        isdir(lock) && rm(lock; recursive = true)
    end
end

"""
    _normalize_manifest_lists!(manifest)

TOML.jl types an empty array from a parsed file as `Vector{Union{}}` on Julia 1.10
(older Base TOML.jl), whereas newer TOML.jl releases (Julia 1.12+) type it as
`Vector{Any}`. A `Vector{Union{}}` can never accept a `push!`, since `Union{}` has no
instances, so an empty `chunks` or `completed_degrees` array parsed back from disk
would make the very first append fail depending solely on the Julia version in use.
Call this right after every `TOML.parsefile` to make the manifest's list fields
concrete and writable regardless of which TOML.jl version produced them.
"""
function _normalize_manifest_lists!(manifest)
    manifest["chunks"] = Vector{Any}(get(manifest, "chunks", Any[]))
    manifest["completed_degrees"] = Vector{Int}(get(manifest, "completed_degrees", Int[]))
    return manifest
end

function _open_checkpoint(options::CheckpointOptions, fingerprint, metadata)
    root = options.path
    isfile(root) && throw(ArgumentError(
        "checkpoint path $root is a legacy file; migrate it to the directory format first"))
    mkpath(joinpath(root, "chunks"))
    path = _manifest_path(root)
    !options.resume && isfile(path) &&
        throw(ArgumentError(
            "checkpoint $root already exists and resume=false; choose a new directory"))
    manifest = if isfile(path)
        parsed = _normalize_manifest_lists!(TOML.parsefile(path))
        get(parsed, "schema_version", 0) == _CHECKPOINT_SCHEMA || throw(ArgumentError(
            "checkpoint $root uses an incompatible schema"))
        get(parsed, "fingerprint", "") == fingerprint || throw(ArgumentError(
            "checkpoint $root does not match this cohomological problem"))
        get(parsed, "problem_id", "") == options.problem_id || throw(ArgumentError(
            "checkpoint $root has a different problem_id"))
        parsed
    else
        Dict{String, Any}(
            "schema_version" => _CHECKPOINT_SCHEMA,
            "problem_id" => options.problem_id,
            "fingerprint" => fingerprint,
            "byte_order" => string(Base.ENDIAN_BOM),
            "metadata" => metadata,
            "completed_degrees" => Int[],
            "chunks" => Any[]
        )
    end
    string(get(manifest, "byte_order", "")) == string(Base.ENDIAN_BOM) ||
        throw(ArgumentError("checkpoint $root was written with a different byte order"))
    isfile(path) || _with_checkpoint_lock(root) do
        _atomic_manifest(path, manifest)
    end
    return CheckpointSession(options, fingerprint, manifest)
end

function _file_sha256(path)
    open(path, "r") do io
        return bytes2hex(SHA.sha256(io))
    end
end

function _write_chunk!(session::CheckpointSession, W, R, degree::Int, indices,
        sparse_solver; degree_complete::Bool)
    isempty(indices) && return nothing
    root = session.options.path
    return _with_checkpoint_lock(root) do
        ids = collect(Int, indices)
        id_hash = bytes2hex(SHA.sha256(reinterpret(UInt8, Int64.(ids))))[1:16]
        filename = "degree_$(lpad(degree, 3, '0'))_$id_hash.bin"
        final_path = joinpath(root, "chunks", filename)
        temporary = final_path * ".tmp.$(getpid())"
        w_slice = Array(view(W.poly.coefficients, :, :, ids))
        r_slice = Array(view(R.poly.coefficients, :, ids))
        checksum = ""
        try
            open(temporary, "w") do io
                write(io, _CHECKPOINT_MAGIC)
                write(io, w_slice)
                write(io, r_slice)
                flush(io)
            end
            checksum = _file_sha256(temporary)
            mv(temporary, final_path; force = true)
        finally
            isfile(temporary) && rm(temporary)
        end
        entry = Dict{String, Any}(
            "file" => filename,
            "sha256" => checksum,
            "degree" => degree,
            "indices" => ids,
            "w_size" => collect(size(w_slice)),
            "r_size" => collect(size(r_slice))
        )
        manifest = _normalize_manifest_lists!(TOML.parsefile(_manifest_path(root)))
        manifest["fingerprint"] == session.fingerprint || throw(ArgumentError(
            "checkpoint fingerprint changed while writing"))
        chunks = manifest["chunks"]
        existing = findfirst(chunk -> chunk["file"] == filename, chunks)
        isnothing(existing) ? push!(chunks, entry) : (chunks[existing] = entry)
        if degree_complete
            degrees = manifest["completed_degrees"]
            degree in degrees || push!(degrees, degree)
            sort!(degrees)
        end
        diagnostics = isnothing(sparse_solver) ? Dict("backend" => "dense") :
                      Dict(
            "backend" => string(_backend_name(sparse_solver.backend)),
            "max_backward_error" => sparse_solver.max_relative_residual,
            "refinement_count" => sparse_solver.refinement_count
        )
        manifest["diagnostics"] = diagnostics
        _atomic_manifest(_manifest_path(root), manifest)
        session.manifest = manifest
        return nothing
    end
end

function _mark_degree_complete!(session::CheckpointSession, degree::Int)
    root = session.options.path
    _with_checkpoint_lock(root) do
        manifest = _normalize_manifest_lists!(TOML.parsefile(_manifest_path(root)))
        manifest["fingerprint"] == session.fingerprint || throw(ArgumentError(
            "checkpoint fingerprint changed while committing degree $degree"))
        degrees = manifest["completed_degrees"]
        degree in degrees || push!(degrees, degree)
        sort!(degrees)
        _atomic_manifest(_manifest_path(root), manifest)
        session.manifest = manifest
    end
    return nothing
end

function _restore_checkpoint!(session::CheckpointSession, W, R;
        verify_existing::Bool = false)
    completed = BitSet()
    for chunk in get(session.manifest, "chunks", Any[])
        path = joinpath(session.options.path, "chunks", chunk["file"])
        isfile(path) || throw(ArgumentError("checkpoint chunk $path is missing"))
        _file_sha256(path) == chunk["sha256"] || throw(ArgumentError(
            "checkpoint chunk $path failed its SHA-256 check"))
        ids = Int.(chunk["indices"])
        w_size = Tuple(Int.(chunk["w_size"]))
        r_size = Tuple(Int.(chunk["r_size"]))
        w_slice = Array{eltype(W.poly.coefficients)}(undef, w_size)
        r_slice = Array{eltype(R.poly.coefficients)}(undef, r_size)
        open(path, "r") do io
            read(io, length(_CHECKPOINT_MAGIC)) == _CHECKPOINT_MAGIC || throw(ArgumentError(
                "checkpoint chunk $path has an invalid header"))
            read!(io, w_slice)
            read!(io, r_slice)
            eof(io) || throw(ArgumentError("checkpoint chunk $path has trailing data"))
        end
        if verify_existing &&
           (view(W.poly.coefficients, :, :, ids) != w_slice ||
            view(R.poly.coefficients, :, ids) != r_slice)
            throw(ArgumentError(
                "initial_solution disagrees with committed checkpoint coefficients in $path"))
        end
        W.poly.coefficients[:, :, ids] .= w_slice
        R.poly.coefficients[:, ids] .= r_slice
        union!(completed, ids)
    end
    return completed
end

function _completed_degrees(session::CheckpointSession)
    Int.(get(session.manifest, "completed_degrees", Int[]))
end
