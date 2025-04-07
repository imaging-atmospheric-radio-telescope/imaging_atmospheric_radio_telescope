Can a 23m, 10Ghz imaing radio telescope see a 10TeV air shower?
===============================================================
Sebastian A. Mueller, 2025-04-07


We agree that a 10TeV air shower indces an inrushing electric field that
spikes at about

E = 10e-6 V m^{-1}

on ground for the frequency range of ASTRA universal which spans 9.5GHz to
10.5GHz and thus has a bandwidth of

B = 1GHz

Assuming we are in the far field and the shower is a point like source this
gives a pointing vector of magnitude, and thus areal power density of:

S = (E ** 2) / Z0

S = 2.65e-13 W m^{-2}.

Where Z0 is the vacuum impedance of 377 Ohm.

The effective antenna area of an ASTRA low noise block at 10GHz is about

A_eff = (lambda ** 2) / (4 * PI)

A_eff = 7.15e-5 m^{2}.

Here lambda is 3e8 m s^{-1} / 10Ghz = 0.03 m.

Now assuming we build a large 23m diameter mirror with A_mirror = 415m^{2}
surface area we should manage to have a gain of at least

G = 1e6.

A_mirror / A_eff is approx 5.8e6, so even with losses in the feed horns, a gain
of 1e6 should be possible.

An inexpensive commercial ASTRA universal LNB has a noise temperature of about

T_antenna = 100K

without active cooling. This gives a noise power in the LNB of

P_thermal_noise_in_lnb = T_antenna * kB * B

P_thermal_noise_in_lnb = 1.4e-12 W

With kB is the Boltzmann constant.

The signal power, in the moment of the inrushing radio flash in LNB should be

P_signal_in_lnb = S * G * A_eff

P_signal_in_lnb = 1.9e-11 W

So I assume that in the brief moment of the radio flash the signal exceeds the
thermal noise by a factor of

P_signal_in_lnb / P_thermal_noise_in_lnb ~= 13.7

And this shoud be easy to find. I would expect that we could still trigger if
P_signal_in_lnb was in the same order of magnitude as P_thermal_noise_in_lnb
because of spatial relations in neighboring pixels.

Now this raises the question: What is wrong with this estimate or why are we not
building such telescopes?
