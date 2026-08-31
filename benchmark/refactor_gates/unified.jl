using MORFE, LinearAlgebra, StaticArrays, Printf
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting, SpectralData
using MORFE.Resonance: ResonanceConfig

B0=[2.0 -1.0; -1.0 2.0];
B2=[1.0 0.0; 0.0 1.0];
B1=0.001*B2
cub = MultilinearMap((res, x1, x2, x3)->(@. res += -1.0*x1*x2*x3), (3, 0))
model = NthOrderModel((B0, B1, B2), (cub,))
sp = spectrum(model; solver = DefaultEigensolver())
select_master_modes_by_sorting(sp, 2)
idx = findall(sp.master_modes)

fails = 0
function chk(n, c)
    (global fails; c || (fails+=1); @printf("  %-50s %s\n", n, c ? "ok" : "MISMATCH"))
end

# OLD three-arg path (model, order, spectrum) vs NEW (model, sd, order)
Wo, Ro = parametrise(model, 5, sp; resonance = :complex_normal_form, resonance_tol = 0.05)
sd = SpectralData(model, sp; master = idx)
Wn, Rn = parametrise(model, sd, 5;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false))
chk("unified parametrise ≡ old entry point (W)", Wo.poly.coefficients ==
                                                 Wn.poly.coefficients)
chk("unified parametrise ≡ old entry point (R)", Ro.poly.coefficients ==
                                                 Rn.poly.coefficients)

# MultiindexSet as expansion_order
mset = all_multiindices_up_to(2, 5; min_degree = 1)
Wm, Rm = parametrise(model, sd, mset;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false))
chk("expansion_order::MultiindexSet ≡ Integer", Wm.poly.coefficients ==
                                                Wn.poly.coefficients)

# conjugate symmetry carried by the bundle
sdc = SpectralData(model, sp; master = idx, conjugate_permutation = :detect)
Wc, Rc = parametrise(model, sdc, 5;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false))
Wr, Rr = parametrise(model, 5, sp; resonance = :complex_normal_form, resonance_tol = 0.05,
    conjugate_permutation = [2, 1])
chk("conjugate symmetry from bundle ≡ explicit literal", Wc.poly.coefficients ==
                                                         Wr.poly.coefficients)

# bad expansion order -> readable error, not MethodError
try
    parametrise(model, sd, "five")
    chk("bad expansion_order errors", false)
catch e
    chk("bad expansion_order -> ArgumentError", e isa ArgumentError)
end

println(fails == 0 ? "\nUNIFIED ENTRY POINT PASSED" : "\nFAILED ($fails)")
fails == 0 || exit(1)
