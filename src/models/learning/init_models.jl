LKDExp(simtime, dictionary, folder, root; interval = 5000, id::String = "") = ExpParams(
    simtime = simtime,
    neurons = 5000,
    types = pv_only,
    conn = lkd2014_soma,
    name = "LKD",
    model = "soma",
    input = "soma",
    synapses = LKDSynapses,
    interval = interval,
    strength = 2.78f0,
    dictionary = dict_name(dictionary),
    dictionary_path = dictionary,
    learn_istdp = lkd_istdp,
    learn_stdp = lkd_stdp,
    root = root,
    id = id,
    folder = folder,
)

DuarteExp(simtime, dictionary, folder, root; interval = 5000, id::String = "", kwargs...) =
    ExpParams(
        simtime = simtime,
        types = duarte_types,
        conn = duartemorrison2017_soma,
        name = "Duarte",
        model = "soma",
        input = "soma",
        synapses = DuarteSynapses,
        interval = interval,
        strength = 1.28f0,
        rate = 3.0f0,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        learn_istdp = duarte_istdp,
        learn_stdp = duarte_stdp,
        root = root,
        id = id,
        folder = folder;
        kwargs...,
    )

TripodExp_asymm(
    simtime,
    dictionary,
    folder,
    root;
    interval = 5000,
    id::String = "",
    kwargs...,
) = ExpParams(
    simtime = simtime,
    types = TNN.duarte_types,
    conn = TNN.lkd2014_dend,
    name = "TripodAsymmetric",
    model = "random",
    input = "asymmetric",
    synapses = TNN.TripodSynapses,
    interval = interval,
    dictionary = dict_name(dictionary),
    dictionary_path = dictionary,
    root = root,
    id = id,
    folder = folder;
    kwargs...,
)

TripodExp_symm(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodCoupledSymmetric",
        model = "random_symm",
        input = "symmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_coupled(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodCoupled",
        model = "random",
        input = "symmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_400(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodCoupled400",
        model = "400;400",
        input = "asymmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_300(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodCoupled300",
        model = "300;300",
        input = "asymmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_150(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodCoupled150",
        model = "150;150",
        input = "asymmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_AMPA(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodAMPA",
        model = "random",
        input = "asymmetric",
        synapses = TNN.TripodSynapses_AMPA,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_NAR(
    simtime,
    dictionary,
    folder,
    root;
    NAR = 1.0,
    interval = 5000,
    id::String = "",
) = ExpParams(
    simtime = simtime,
    types = TNN.duarte_types,
    conn = TNN.lkd2014_dend,
    name = "TripodNAR",
    model = "random",
    input = "asymmetric",
    synapses = TNN.TripodSynapses_NAR(; NAR = Float32(NAR)),
    interval = interval,
    dictionary = dict_name(dictionary),
    dictionary_path = dictionary,
    root = root,
    id = id,
    folder = folder,
)

TripodExp_NMDA_short(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodNMDA_short",
        model = "random",
        input = "asymmetric",
        synapses = TNN.TripodSynapses_NMDA_short,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )

TripodExp_single(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodSingleDendrite",
        model = "single",
        input = "symmetric",
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        exc_noise = 2.0,
        folder = folder,
    )

TripodExp_single_(simtime, dictionary, folder, root; interval = 5000, id::String = "") =
    ExpParams(
        simtime = simtime,
        types = TNN.duarte_types,
        conn = TNN.lkd2014_dend,
        name = "TripodSingleDendrite_denser",
        model = "single",
        input = "symmetric",
        density = 1e-1,
        synapses = TNN.TripodSynapses,
        interval = interval,
        dictionary = dict_name(dictionary),
        dictionary_path = dictionary,
        root = root,
        id = id,
        folder = folder,
    )



init_models = collect(
    Set([
        TripodExp_single,
        TripodExp_NMDA_short,
        TripodExp_NAR,
        TripodExp_AMPA,
        TripodExp_150,
        TripodExp_300,
        TripodExp_400,
        TripodExp_symm,
        TripodExp_coupled,
        TripodExp_asymm,
        DuarteExp,
        LKDExp,
    ]),
)

export TripodExp_150,
    TripodExp_300,
    TripodExp_400,
    TripodExp_symm,
    TripodExp_coupled,
    TripodExp_asymm,
    DuarteExp,
    LKDExp,
    TripodExp_AMPA,
    TripodExp_NAR,
    TripodExp_NMDA_short,
    TripodExp_single,
    TripodExp_single_,
    init_models