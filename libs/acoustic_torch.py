import torch
import drjit as dr


@dr.wrap_ad(source='drjit', target='torch')
def EDC(rir, energy=True, db=True, norm=True):
    rir = rir if energy else torch.square(rir)
    sch = torch.flip(torch.cumsum(torch.flip(rir, [0]), dim=0), [0])
    sch = sch / torch.sum(rir, 0) if norm else sch
    return 10. * torch.log10(sch) if db   else sch


@dr.wrap_ad(source='drjit', target='torch')
def T(time, edc, dB_init=-5., dB_decay=30.):
    assert len(time.shape) == 1
    assert len(edc.shape)  == 2
    assert time.shape[0]   == edc.shape[0]
    rt60 = torch.zeros(edc.shape[1], device=edc.device) * torch.nan

    for k in range(edc.shape[1]):
        i = torch.masked.argmax(edc[:, k], dtype=torch.int, mask=(edc[:, k] < dB_init)).item()
        j = torch.masked.argmax(edc[:, k], dtype=torch.int, mask=(edc[:, k] < dB_init - dB_decay)).item()

        if j <= i or j == edc.shape[0]:
            continue

        X = torch.stack([time[i:j+1], torch.ones(j-i+1, device=edc.device)])
        rt60[k] = -60. / (torch.inverse(X @ X.T) @ X @ edc[i:j+1, k])[0]

    return rt60


@dr.wrap_ad(source='drjit', target='torch')
def C(rir, te=80, fs=1000, energy=True):
    ti = int((te / 1000) * fs + 1.)
    rir = rir if energy else torch.square(rir)
    E_bef = torch.sum(rir[:ti], dim=0)
    E_aft = torch.sum(rir[ti:], dim=0)
    return 10. * torch.log10(E_bef / E_aft)


@dr.wrap_ad(source='drjit', target='torch')
def D(rir, fs=1000, energy=True):
    ti = int(0.050 * fs + 1.)
    rir = rir if energy else torch.square(rir)
    E_bef = torch.sum(rir[:ti], dim=0)
    E_aft = torch.sum(rir     , dim=0)
    return E_bef / E_aft


@dr.wrap_ad(source='drjit', target='torch')
def TS(time, rir, energy=True):
    rir_t = (rir.T * time).T
    if not energy:
        rir_t = torch.square(rir_t)
        rir   = torch.square(rir)
    E_bef = torch.sum(rir_t, dim=0)
    E_aft = torch.sum(rir  , dim=0)
    return E_bef / E_aft
