import cupy as cp

def sigma_clip_cupy(data, sigma=3.0, maxiters=5, cenfunc=cp.mean, stdfunc=cp.std):
    data = cp.asarray(data)
    mask = cp.ones_like(data, dtype=bool)

    for _ in range(maxiters):
        subset = data[mask]
        center = cenfunc(subset)
        std = stdfunc(subset)
        lower, upper = center - sigma * std, center + sigma * std

        new_mask = (data >= lower) & (data <= upper)
        if cp.all(mask == new_mask):
            break
        mask = new_mask

    return data[mask], lower, upper


raw_data = cp.array([1, 2, 2, 3, 3, 100, 3, 2, -50])
clipped_data, low, high = sigma_clip_cupy(raw_data)

print("Clipped:", clipped_data)
print("Lower:", low, "Upper:", high)
