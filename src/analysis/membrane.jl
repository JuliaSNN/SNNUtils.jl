function score_membrane(; path, tt0::Int = 0)
    @show path
    @assert(isdir(path))

    class_path = joinpath(path, "analysis")
    !(isdir(class_path)) && (mkdir(class_path))
    save_path = joinpath(class_path, "logreg_membrane.h5")
    isfile(save_path) && rm(save_path)

    stim, seq, net, (dend1, dend2), learn, store = read_network_params(path)

    ## Prepare features
    ## ================

    ## Retrieve features and average points within the stimulus interval
    states = read_network_states(store.data, tt0 = tt0)
    read.(states)
    features, n_feat = states_to_features(states, average_points = true)
    @show n_feat

    idxs = rescale_with_tt0_ratio(store.data, stim, tt0)
    score_features(;
        features = features,
        n_neurons = n_feat,
        seq = seq,
        idxs = idxs,
        save_path = save_path,
        feats_set = (;
            :currents => (n_feat+1):(2*n_feat),
            :membrane => 1:n_feat,
            :all => 1:(2*n_feat),
        ),
    )
end