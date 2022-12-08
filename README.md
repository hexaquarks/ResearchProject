## Simulation and analysis of diffusion of cell membrane proteins in different confinement models.

>The manuscript can be viewed [here](https://drive.google.com/file/d/1yzX9_J3ymIe8_5ZaWhY37LsGaKtz8yi1/view?usp=sharing)

### Note
This is my undergraduate research project that was done for PHYS 449 at McGill. It was in supervision of Dr. Paul Wiseman.

### The simulations
The confinement models simulated are
<ol type="a">
  <li>No confinement (free diffusion)</li>
  <li>Lattice of variable size (from underlying cytoskeleton of cell membrane)</li>
  <li>Transient diffusion in nanoscale circular domains </li>
</ol>

<img src="https://user-images.githubusercontent.com/25794626/206052983-d8a5b5b1-3744-4d49-af38-c3a51165f271.png" width="550">

### The goal and short theoretical description
The goal was to compare STICS and iMSD image fluctuation anaylses in extracting diffusion coefficients of a simulated environment.
STICS method relies on fluctuation of fluorescence intensity in pixels of an image that models microscopy where

$$
G(\zeta, \eta, \tau) = \frac{\langle i(x,y,t) \cdot i(x + \zeta, y + \eta, t + \tau) \rangle}{\langle i(x,y,t) \rangle^2} - 1 =  \frac{\mathcal{F}^{-1} \{ \mathcal{F}(i_{2D}) \cdot \mathcal{F}^{\ast} (i_{2D})\}}{\langle i(x,y,t) \rangle^2}-1,
$$

for which $G(\zeta, \eta, \tau)$ is a approximated by a Gaussian surface. By fitting this Gaussian surface there exists a relationship between the diffusion coeffient of the system and the amplitude decay of the correlation function.

In the iMSD method, the growth of the radius of $G(\zeta, \eta, \tau)$ as a function of time lag $\tau$ can be found through 

$$
(\zeta_e, \eta_e) = \left(\sqrt{-2 \ln\left(\frac{1}{e} - \frac{g_0}{g}\right)\sigma_\zeta^2} + \zeta_0 , \eta_0\right),
$$

with $\sigma_r(\tau_i) = |\zeta_e - \zeta_0|$ as the radius at time $\tau_i$. $\zeta_r(\tau)$ is related to the diffusion coefficient of the system through 

$$
\sigma_r^2(\tau) \approxeq \frac{L^2}{3} \left( 1 - \exp{-\frac{\tau}{\tau_c}}\right) +4D\tau+ \sigma_0^2.
$$

