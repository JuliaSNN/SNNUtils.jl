# Clopath 2010
@with_kw struct STDP
    #voltage based stdp
    a⁻::Float32 = 0.0f0    #ltd strength (pF/mV) # a*(V-θ) = weight
    a⁺::Float32 = 0.0f0    #ltp strength (pF/mV)
    θ⁻::Float32 = -90.0f0 #ltd voltage threshold (mV)
    θ⁺::Float32 = 0.0f0 #ltp voltage threshold (mV)
    τu::Float32 = 1.0f0  #timescale for u variable   (1/ms)
    τv::Float32 = 1.0f0  #timescale for v variable   (1/ms)
    τx::Float32 = 1.0f0  #timescale for x variable   (1/ms)
    τ1::Float32 = 1.0f0
    ϵ::Float32 = 1.0f0  # filter for delayed membrane potential.
    j⁻::Float32 = 0.0f0 # minimum weight
    j⁺::Float32 = 100.0f0 # maximum weight
    τu⁻::Float32 = 1 / τu  #timescale for u variable   (1/ms)
    τv⁻::Float32 = 1 / τv  #timescale for v variable   (1/ms)
    τx⁻::Float32 = 1 / τx  #timescale for x variable   (1/ms)
    τ1⁻::Float32 = 1 / τ1

end

#Vogel 2011
#inhibitory stdp
struct ISTDP
    τy::Float32
    η::Float32
    r0::Float32
    α::Float32
    j⁻::Float32  # minimum weight
    j⁺::Float32  # maximum weight
    # τy⁻::Float32
end

struct TripletRule
    A⁺₂::Float32
    A⁺₃::Float32
    A⁻₂::Float32
    A⁻₃::Float32
    τˣ::Float32
    τʸ::Float32
    τ⁺::Float32
    τ⁻::Float32
    # τˣ⁻::Float32
    # τʸ⁻::Float32
    # τ⁺⁻::Float32
    # τ⁻⁻::Float32
end

# Gutig 2003
struct NLTAH
    τ::Float32
    λ::Float32
    μ::Float32
end

export NLTAH, ISTDP, STDP, TripletRule