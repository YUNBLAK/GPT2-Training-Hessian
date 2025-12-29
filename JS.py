import json
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_js_blockwise(
    eigen_file,
    out_png="js_distance_blockwise_hessian.png",
    num_bins=200,
):
    with open(eigen_file, "r") as f:
        eig_dict = json.load(f)

    layer_names = list(eig_dict.keys())
    vals_list = [np.array(eig_dict[name], dtype=np.float64) for name in layer_names]
    all_vals = np.concatenate(vals_list)
    bins = np.linspace(all_vals.min(), all_vals.max(), num_bins + 1)

    def spectrum_to_density(v):
        hist, _ = np.histogram(v, bins=bins, density=True)
        p = hist.astype(np.float64)
        return p  

    densities = [spectrum_to_density(v) for v in vals_list]
    def kl_div(p, q, eps=1e-12):
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        return np.sum(p * np.log(p / q))

    def js_distance(p, q):
        m = 0.5 * (p + q)
        js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return js

    L = len(densities)
    js_mat = np.zeros((L, L), dtype=np.float64)

    for i in range(L):
        for j in range(i + 1): 
            d = js_distance(densities[i], densities[j])
            js_mat[i, j] = d
            js_mat[j, i] = d

    if L > 1:
        tril_idx = np.tril_indices(L, k=-1)
        js0 = js_mat[tril_idx].mean()
    else:
        js0 = 0.0

    print("max JS value:", js_mat.max())
    print("JS0 (average pairwise JS, lower triangle):", js0)

    mask = np.triu(np.ones_like(js_mat, dtype=bool), k=1)
    js_masked = np.ma.array(js_mat, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(js_masked, cmap="coolwarm", vmin=0, vmax=3)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(layer_names, rotation=90)
    ax.set_yticklabels(layer_names)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("JS distance (unnormalized)")

    plt.title("The JS distance among blockwise Hessian spectra")
    plt.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    return js_mat, js0, layer_names


# if __name__ == "__main__":
#     eigen_file = "files/grad1_layer_4_batch8_init_minibatch_True_bs_8_m_100_v_10_ckpt_0/layer_eigenvalues.json"
#     out_png = "js_distance_blockwise_hessian_init.png"

#     js_mat, js0, layer_names = compute_js_blockwise(eigen_file, out_png=out_png)