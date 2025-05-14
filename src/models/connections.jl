
duarte_types = [0.8, 0.2 * 0.65, 0.2 * 0.35]
pv_only = [0.8, 0.0, 0.2]
sst_only = [0.8, 0.2, 0.0]


duartemorrison2017_dend = (
    EsE = (p = 0.168, μ = 0.45, dist = Normal, σ = 1),
    E_to_If = (p = 0.575, μ = 1.65, dist = LogNormal),
    E_to_Is = (p = 0.244, μ = 0.638, dist = LogNormal),
    If_to_E = (p = 0.60, μ = 5.148, dist = LogNormal),
    EsIs = (p = 0.465, μ = 4.85, dist = LogNormal),
    If_to_If = (p = 0.55, μ = 2.22, dist = LogNormal),
    If_to_Is = (p = 0.24, μ = 1.4, dist = LogNormal),
    Is_to_If = (p = 0.379, μ = 1.47, dist = LogNormal),
    Is_to_Is = (p = 0.381, μ = 0.83, dist = LogNormal)
)

lkd2014_dend = (
    E_to_Ed = (p = 0.2, μ = 15.78, dist = Normal),
    E_to_Is = (p = 0.2, μ = 1.27, dist = LogNormal, σ=0),
    E_to_If = (p = 0.2, μ = 0.27, dist = LogNormal, σ=0),
    If_to_E = (p = 0.2, μ = 15.8,),
    Is_to_Ed = (p = 0.2, μ = 15.8,),
    If_to_If = (p = 0.2, μ = 16.2, ),
    If_to_Is = (p = 0.2, μ = 1.4, ),
    Is_to_If = (p = 0.2, μ = 0.83 ),
    Is_to_Is = (p = 0.2, μ = 0.83,)
)

quaresima2023_dend= (
    # E_to_Ed = (p = 0.0,  μ = 1.78, dist = Normal, σ = 1),
    E_to_Ed = (p = 0.168,  μ = 2.8, dist = Normal, σ = 1),
    E_to_If = (p = 0.2,  μ = log(1.0), dist = LogNormal, σ = 0.1),
    E_to_Is = (p = 0.2,  μ = log(1.0), dist = LogNormal, σ = 0.1),
    If_to_E = (p = 0.2,  μ = log(15.8), dist = LogNormal, σ = 0),
    If_to_Is = (p = 0.2, μ = log(1.4),  dist = LogNormal, σ = 0.25),
    If_to_If = (p = 0.2, μ = log(16.2), dist = LogNormal, σ = 0.14),
    Is_to_Ed = (p = 0.2, μ = log(15.8), dist = LogNormal, σ = 0),
    Is_to_If = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),
    Is_to_Is = (p = 0.2, μ = log(0.83), dist = LogNormal, σ = 0.),

)
    

lkd2014_dend_upinh = (
    E_to_Ed = (p = 0.2, μ = 10.78, dist = Normal, σ = 1),
    E_to_Is = (p = 0.4, μ = 5.27, dist = LogNormal),
    E_to_If = (p = 0.4, μ = 5.27, dist = LogNormal),
    If_to_E = (p = 0.4, μ = 15.8, dist = LogNormal),
    Is_to_Ed = (p = 0.4, μ = 15.8, dist = LogNormal),
    If_to_If = (p = 0.4, μ = 16.2, dist = LogNormal),
    If_to_Is = (p = 0.4, μ = 1.4, dist = LogNormal),
    Is_to_If = (p = 0.4, μ = 0.83, dist = LogNormal),
    Is_to_Is = (p = 0.4, μ = 0.83, dist = LogNormal)
)

lkd2014_soma = (
    EsE = (p = 0.2, μ = 2.76, dist = LogNormal),
    E_to_Is = (p = 0.2, μ = 1.27, dist = LogNormal),
    E_to_If = (p = 0.2, μ = 1.27, dist = LogNormal),
    If_to_E = (p = 0.2, μ = 48.7, dist = LogNormal),
    If_to_If = (p = 0.2, μ = 16.2, dist = LogNormal)
)

lkd2014_soma_j = j0 -> (
    EsE = (p = 0.2, μ = j0, dist = LogNormal),
    E_to_Is = (p = 0.2, μ = 1.27, dist = LogNormal),
    E_to_If = (p = 0.2, μ = 1.27, dist = LogNormal),
    If_to_E = (p = 0.2, μ = 48.7, dist = LogNormal),
    If_to_If = (p = 0.2, μ = 16.2, dist = LogNormal)
)

duartemorrison2017_soma = (
    EsE = (p = 0.168, μ = 0.45, dist = Normal, σ = 1),
    E_to_If = (p = 0.575, μ = 1.65, dist = LogNormal),
    E_to_Is = (p = 0.244, μ = 0.638, dist = LogNormal),
    If_to_E = (p = 0.60, μ = 5.148, dist = LogNormal),
    EsIs = (p = 0.465, μ = 4.85, dist = LogNormal),
    If_to_If = (p = 0.55, μ = 2.22, dist = LogNormal),
    If_to_Is = (p = 0.24, μ = 1.4, dist = LogNormal),
    Is_to_If = (p = 0.379, μ = 1.47, dist = LogNormal),
    Is_to_Is = (p = 0.381, μ = 0.83, dist = LogNormal)
)

no_connections = (
    EsE = (p = 0.0, μ = 0.0, dist = LogNormal),
    E_to_If = (p = 0.0, μ = 0.0, dist = LogNormal),
    E_to_Is = (p = 0.0, μ = 0.0, dist = LogNormal),
    If_to_E = (p = 0.0, μ = 0.0, dist = LogNormal),
    EsIs = (p = 0.0, μ = 0.0, dist = LogNormal),
    If_to_If = (p = 0.0, μ = 0.0, dist = LogNormal),
    If_to_Is = (p = 0.0, μ = 0.0, dist = LogNormal),
    Is_to_If = (p = 0.0, μ = 0.0, dist = LogNormal),
    Is_to_Is = (p = 0.0, μ = 0.0, dist = LogNormal)
)

export duarte_types, pv_only, sst_only, duartemorrison2017_dend, lkd2014_dend, quaresima2023_dend, lkd2014_dend_upinh, lkd2014_soma, lkd2014_soma_j, duartemorrison2017_soma, no_connections