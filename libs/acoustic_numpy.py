import numpy as np


def EDC(rir, energy=True, db=True, norm=True):
    """
    Die Abklingkurve jedes Oktavbands wird durch Rückwärts-Integration der quadrierten Impulsantwort berechnet.
    Im Idealfall ohne Störpegel sollte die Integration am Ende der Impulsantwort beginnen und bis zum
    Anfang der quadrierten Impulsantwort laufen.
    """

    rir = np.array(rir) if energy else np.square(np.array(rir))

    # Schroeder integration
    sch = np.cumsum(rir[::-1], axis=0)[::-1]

    sch = sch / np.sum(rir, axis=0) if norm else sch

    if db:
        with np.errstate(divide="ignore"):
            sch = 10. * np.log10(sch)
    return sch


def T(time, edc, dB_init=-5., dB_decay=30.):
    """
    Zeit, die erforderlich ist, damit die räumlich gemittelte Schallenergiedichte in
    einem geschlossenen Raum um 60 dB sinkt, nachdem die Schallquelle abgeschaltet wurde.

    T kann basierend auf einem kürzeren Dynamikbereich als 60 dB ermittelt und auf eine Abklingzeit von
    60 dB extrapoliert werden. Sie wird dann entsprechend gekennzeichnet. So wird sie, wenn T aus der Zeit ermittelt wird, zu
    der die Abklingkurve erstmalig die Werte 5 dB und 25 dB unter dem Anfangspegel erreicht, mit T 20 , gekennzeichnet.
    Werden Abklingwerte von 5 dB bis 35 dB unter dem Anfangspegel verwendet, werden sie mit T 30 gekennzeichnet.
    """

    time, edc = np.array(time), np.array(edc)
    assert time.shape == edc.shape

    i = np.argmax(edc < dB_init)
    j = np.argmax(edc < dB_init - dB_decay)
    if j <= i:
        return np.NaN

    X = np.stack([time[i:j+1], np.ones(j-i+1)])
    rt60 = -60 / (np.linalg.inv(X @ X.T) @ X @ edc[i:j+1])[0]

    return rt60


def C(te, rir, fs=1000., energy=True):
    """
    C_te: eine Früh-zu-Spät-Index genannte Kennzahl;

    te = Zeitgrenze, i.d.R. entweder 50 ms oder 80 ms (C 80 wird üblicherweise „Klarheitsmaß“ genannt)
    rir = die Impulsantwort
    """

    rir = np.array(rir) if energy else np.square(np.array(rir))
    ti = int((te / 1000.) * fs + 1.)
    E_bef = np.sum(rir[:ti])
    E_aft = np.sum(rir[ti:])
    return 10. * np.log10(E_bef / E_aft)


def D50(rir, fs=1000., energy=True):
    """
    Verhältnis der früh eintreffenden Energie zur Gesamt-Schallenergie („Tonschärfe“ oder „Deutlichkeit“)
    """

    rir = np.array(rir) if energy else np.square(np.array(rir))
    ti = int(0.050 * fs + 1)
    E_bef = np.sum(rir[:ti])
    E_tot = np.sum(rir)
    return E_bef / E_tot


def TS(time, rir, energy=True):
    rir = np.array(rir)
    n = 1 if energy else 2
    return np.sum(np.power(rir * time, n)) / np.sum(np.power(rir, n))
