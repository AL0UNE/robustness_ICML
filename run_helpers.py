from joblib import Parallel, delayed


def _format_run_name(benchmark_test_name=None):
    return benchmark_test_name if benchmark_test_name is not None else "RUN"


def run_cv_parallel(
    task_fn,
    levels,
    models,
    splits,
    n_jobs,
    task_kwargs=None,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run a task across models, perturbation levels, and CV folds."""
    if task_kwargs is None:
        task_kwargs = {}

    call_kwargs = dict(task_kwargs)
    if benchmark_test_name is not None and "preset_test_name" not in call_kwargs:
        call_kwargs["preset_test_name"] = benchmark_test_name

    run_name = _format_run_name(benchmark_test_name)
    n_levels = len(levels)
    n_models = len(models)
    n_splits = len(splits)
    print(
        f"[{run_name}] Starting CV run: {n_levels} level(s), {n_models} model(s), {n_splits} fold(s)",
        flush=True,
    )

    if not split_train_predict:
        results = []
        for level_idx, level in enumerate(levels, start=1):
            print(
                f"[{run_name}] Level {level_idx}/{n_levels} started (level={level})",
                flush=True,
            )
            level_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(task_fn)(
                    model_name,
                    model,
                    level,
                    train_idx,
                    val_idx,
                    fold_idx,
                    **call_kwargs,
                )
                for model_name, model in models.items()
                for fold_idx, (train_idx, val_idx) in enumerate(splits)
            )
            results.extend(level_results)
            print(
                f"[{run_name}] Level {level_idx}/{n_levels} completed",
                flush=True,
            )
        print(f"[{run_name}] CV run completed", flush=True)
        return results

    if evaluate_bundle_fn is None:
        raise ValueError("evaluate_bundle_fn must be provided when split_train_predict=True")

    train_kwargs = dict(call_kwargs)
    train_kwargs["return_trained"] = True
    train_jobs = n_jobs if train_n_jobs is None else train_n_jobs
    eval_kwargs = evaluate_bundle_kwargs or {}

    results = []
    for level_idx, level in enumerate(levels, start=1):
        print(
            f"[{run_name}] Level {level_idx}/{n_levels} training started (level={level})",
            flush=True,
        )
        trained_bundles = Parallel(n_jobs=train_jobs, verbose=verbose)(
            delayed(task_fn)(
                model_name,
                model,
                level,
                train_idx,
                val_idx,
                fold_idx,
                **train_kwargs,
            )
            for model_name, model in models.items()
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        )
        print(
            f"[{run_name}] Level {level_idx}/{n_levels} evaluation started",
            flush=True,
        )
        level_results = Parallel(n_jobs=predict_n_jobs, verbose=verbose)(
            delayed(evaluate_bundle_fn)(bundle, **eval_kwargs) for bundle in trained_bundles
        )
        results.extend(level_results)
        print(
            f"[{run_name}] Level {level_idx}/{n_levels} completed",
            flush=True,
        )

    print(f"[{run_name}] CV run completed", flush=True)
    return results


def run_cv_parallel_and_save(
    task_fn,
    levels,
    directory,
    test_name,
    columns,
    models,
    splits,
    n_jobs,
    n_folds,
    save_results_fn,
    task_kwargs=None,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run a CV task and persist results using the project result schema."""
    effective_test_name = test_name if benchmark_test_name is None else benchmark_test_name
    print(f"[{effective_test_name}] Dispatching CV jobs", flush=True)
    results = run_cv_parallel(
        task_fn,
        levels,
        models,
        splits,
        n_jobs,
        task_kwargs=task_kwargs,
        verbose=verbose,
        split_train_predict=split_train_predict,
        train_n_jobs=train_n_jobs,
        predict_n_jobs=predict_n_jobs,
        evaluate_bundle_fn=evaluate_bundle_fn,
        evaluate_bundle_kwargs=evaluate_bundle_kwargs,
        benchmark_test_name=effective_test_name,
    )
    print(f"[{effective_test_name}] Writing results file", flush=True)
    save_results_fn(results, directory, n_folds, columns=columns, test_name=test_name)
    print(f"[{effective_test_name}] Results file written", flush=True)
    return results


def run_subgroup_parallel(
    task_fn,
    variable,
    models,
    splits,
    n_jobs,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run subgroup tasks across models and folds and flatten outputs."""
    run_name = _format_run_name(benchmark_test_name)
    print(
        f"[{run_name}] Starting subgroup run: {len(models)} model(s), {len(splits)} fold(s)",
        flush=True,
    )
    if not split_train_predict:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(task_fn)(
                model_name,
                model,
                train_idx,
                val_idx,
                variable,
                preset_test_name=benchmark_test_name,
            )
            for model_name, model in models.items()
            for train_idx, val_idx in splits
        )
        flat_results = [x for xs in results if xs is not None for x in xs]
        print(f"[{run_name}] Subgroup run completed", flush=True)
        return flat_results

    if evaluate_bundle_fn is None:
        raise ValueError("evaluate_bundle_fn must be provided when split_train_predict=True")

    train_jobs = n_jobs if train_n_jobs is None else train_n_jobs
    eval_kwargs = evaluate_bundle_kwargs or {}
    print(f"[{run_name}] Subgroup training started", flush=True)
    trained_bundles = Parallel(n_jobs=train_jobs, verbose=verbose)(
        delayed(task_fn)(
            model_name,
            model,
            train_idx,
            val_idx,
            variable,
            return_trained=True,
            preset_test_name=benchmark_test_name,
        )
        for model_name, model in models.items()
        for train_idx, val_idx in splits
    )
    print(f"[{run_name}] Subgroup evaluation started", flush=True)
    results = Parallel(n_jobs=predict_n_jobs, verbose=verbose)(
        delayed(evaluate_bundle_fn)(bundle, **eval_kwargs) for bundle in trained_bundles
    )
    flat_results = [x for xs in results if xs is not None for x in xs]
    print(f"[{run_name}] Subgroup run completed", flush=True)
    return flat_results


def run_subgroup_parallel_and_save(
    task_fn,
    variable,
    directory,
    test_name,
    columns,
    models,
    splits,
    n_jobs,
    n_folds,
    save_results_fn,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run a subgroup task and persist results."""
    effective_test_name = test_name if benchmark_test_name is None else benchmark_test_name
    print(f"[{effective_test_name}] Dispatching subgroup jobs", flush=True)
    results = run_subgroup_parallel(
        task_fn,
        variable,
        models,
        splits,
        n_jobs,
        verbose=verbose,
        split_train_predict=split_train_predict,
        train_n_jobs=train_n_jobs,
        predict_n_jobs=predict_n_jobs,
        evaluate_bundle_fn=evaluate_bundle_fn,
        evaluate_bundle_kwargs=evaluate_bundle_kwargs,
        benchmark_test_name=effective_test_name,
    )
    print(f"[{effective_test_name}] Writing results file", flush=True)
    save_results_fn(results, directory, n_folds, columns=columns, test_name=test_name)
    print(f"[{effective_test_name}] Results file written", flush=True)
    return results


def run_temporal_parallel(
    task_fn,
    X_train,
    y_train,
    df_test,
    models,
    n_jobs,
    stratify_on=None,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run temporal validation across models, optionally stratified and flattened."""
    run_name = _format_run_name(benchmark_test_name)
    split_desc = "overall" if stratify_on is None else f"stratified by {stratify_on}"
    print(
        f"[{run_name}] Starting temporal run ({split_desc}) with {len(models)} model(s)",
        flush=True,
    )
    if not split_train_predict and stratify_on is None:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(task_fn)(
                model_name,
                model,
                X_train,
                y_train,
                df_test,
                stratify_on=None,
                preset_test_name=benchmark_test_name,
            )
            for model_name, model in models.items()
        )
        print(f"[{run_name}] Temporal run completed", flush=True)
        return results

    if not split_train_predict:
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(task_fn)(
                model_name,
                model,
                X_train,
                y_train,
                df_test,
                stratify_on=stratify_on,
                preset_test_name=benchmark_test_name,
            )
            for model_name, model in models.items()
        )
        flat_results = [x for xs in results if xs is not None for x in xs]
        print(f"[{run_name}] Temporal run completed", flush=True)
        return flat_results

    if evaluate_bundle_fn is None:
        raise ValueError("evaluate_bundle_fn must be provided when split_train_predict=True")

    train_jobs = n_jobs if train_n_jobs is None else train_n_jobs
    eval_kwargs = evaluate_bundle_kwargs or {}
    print(f"[{run_name}] Temporal training started", flush=True)
    trained_bundles = Parallel(n_jobs=train_jobs, verbose=verbose)(
        delayed(task_fn)(
            model_name,
            model,
            X_train,
            y_train,
            df_test,
            stratify_on=stratify_on,
            return_trained=True,
            preset_test_name=benchmark_test_name,
        )
        for model_name, model in models.items()
    )
    print(f"[{run_name}] Temporal evaluation started", flush=True)
    results = Parallel(n_jobs=predict_n_jobs, verbose=verbose)(
        delayed(evaluate_bundle_fn)(bundle, **eval_kwargs) for bundle in trained_bundles
    )

    if stratify_on is None:
        print(f"[{run_name}] Temporal run completed", flush=True)
        return results
    flat_results = [x for xs in results if xs is not None for x in xs]
    print(f"[{run_name}] Temporal run completed", flush=True)
    return flat_results


def run_temporal_parallel_and_save(
    task_fn,
    X_train,
    y_train,
    df_test,
    directory,
    test_name,
    columns,
    stratify_on,
    models,
    n_jobs,
    n_folds,
    save_results_fn,
    verbose=0,
    split_train_predict=False,
    train_n_jobs=None,
    predict_n_jobs=1,
    evaluate_bundle_fn=None,
    evaluate_bundle_kwargs=None,
    benchmark_test_name=None,
):
    """Run a stratified temporal task and persist results."""
    effective_test_name = test_name if benchmark_test_name is None else benchmark_test_name
    print(f"[{effective_test_name}] Dispatching temporal jobs", flush=True)
    results = run_temporal_parallel(
        task_fn,
        X_train,
        y_train,
        df_test,
        models,
        n_jobs,
        stratify_on=stratify_on,
        verbose=verbose,
        split_train_predict=split_train_predict,
        train_n_jobs=train_n_jobs,
        predict_n_jobs=predict_n_jobs,
        evaluate_bundle_fn=evaluate_bundle_fn,
        evaluate_bundle_kwargs=evaluate_bundle_kwargs,
        benchmark_test_name=effective_test_name,
    )
    print(f"[{effective_test_name}] Writing results file", flush=True)
    save_results_fn(results, directory, n_folds, columns=columns, test_name=test_name)
    print(f"[{effective_test_name}] Results file written", flush=True)
    return results
