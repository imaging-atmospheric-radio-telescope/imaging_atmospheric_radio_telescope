Sebastian A. Mueller,
Christian Monstein,
Adrian Biland,
Axel Arbet-Engels

ETH Zurich

- Sebastian presents the imaging atmospheric arkaryan telescope to Christian. He points out the similarity to the existing IACTs.

- Christian operates three radio telescopes for ETH and is alwas looking for future challenges for these instruments. Adrian and Sebastian explain how a hybrid detection with a classic radio telescope and an IACT might be performed on the existing radio telescopes of ETH.

5m dish Bleien: http://soleil.i4ds.ch/solarradio/data/status/RSG/status5m.php
5m dish Zürich: http://soleil.i4ds.ch/solarradio/data/status/STS/statusSTS_nf.php
7m dish Bleien: http://soleil.i4ds.ch/solarradio/data/status/RSG/status7m.php


- Christian would like to see the signals simulated by Sebastian in units which are more common in radio astronomy. He presents a webpage which he is familiar with and explains the basic conversions of signal representations in radio astronomy. https://www.craf.eu/useful-equations/conversion-formulae/.

- Christian is briefly presented the image formation algorithm implemented by Sebastian and after a short discussion about the possibility of superpositioning of electric field strengths, he does not see a fundamental problem. Also the gain of the imaging reflector was discussed and Sebastian's approach based on the areal ratio of reflector dish and pixel antenna seems to be a valid approach.

- Based on the typical antenna pixel electrical field strength amplitudes which Christian found in the simulation figures presented by Sebastian, he estimated the air-shower signal strength compared to the expected noise of antennas. He provided a tiny python script:

```python
# Copyright Christian Monstein 2017

import numpy as np

E   = 40e-6    # maximum fieldstrenght taken from simulation [V/m]
Zo  = 377.0    # free space impedance 120 pi [ohm]
B   = 20e6     # assumption bandwidth [Hz]
k   = 1.38e-23 # Boltzmann constant [J/K]
Trx = 80.0     # 'normal' frontend with low noise amplifier at ambient temperature [K]
Ti  = 1e-10    # integration time [s]

P = E**2 / Zo
print('Expected power at dipole antenna {:4.1e} Watt'.format(P))

Ta = P/(k*B)
print('Expected antenna temperature {:.0f} Kelvin'.format(Ta))

Tsys = Ta + Trx
Y = Tsys/Trx
print('Expected y-factor {:.1f} dB'.format(10.0*np.log10(Y)))

dT = Trx / (np.sqrt(B*Ti))
print('Temperature resolution {:.0f} Kelvin'.format(dT))

snr = Ta/dT
print('Expected SNR {:.1f} dB'.format(10.0*np.log10(snr)))
```

Output:
```
Expected power at dipole antenna 4.2e-12 Watt
Expected antenna temperature 15377 Kelvin
Expected y-factor 22.9 dB
Temperature resolution 1789 Kelvin
Expected SNR 9.3 dB
```

Christian's comment on this:
```
Hi Sebastian

A quick check shows me that it should be doable with simple rf-electronics without LN2-cooling, see script below.
I expect an SNR of > 9dB (~8x)

Cheers,
Christian
```

- Sebastian concludes that we need more crosschecks of Sebastian's simulation tool. After all, this is the first time Sebastian simulates a radio instrument.

