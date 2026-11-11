"""Bilingual (es/en) texts of the report, in LaTeX.

The markers ``<<name>>`` are replaced with :func:`fill`. This way the texts
can contain LaTeX braces without escaping them.
"""

from __future__ import annotations

import re


def fill(text: str, **kw) -> str:
    def rep(m):
        k = m.group(1)
        if k not in kw:
            raise KeyError(f"Falta el valor '{k}' para el texto")
        return str(kw[k])
    return re.sub(r"<<(\w+)>>", rep, text)


STATUS = {
    "good": ("Bien", "Good"),
    "caution": ("Revisar", "Caution"),
    "poor": ("Problema", "Poor"),
    "info": ("Info", "Info"),
}

# key: (label_es, label_en, criterion_es, criterion_en, advice_es, advice_en)
ASSESS = {
    "imbalance": (
        r"Desequilibrio de flujo $|\sum B_z|/\sum|B_z|$",
        r"Flux imbalance $|\sum B_z|/\sum|B_z|$",
        r"$<10\%$ bien; $10$--$20\%$ revisar",
        r"$<10\%$ good; $10$--$20\%$ caution",
        r"Parte del flujo se cierra fuera del dominio. Amplía el recorte para incluir ambas polaridades completas o usa \texttt{balance\_flux()} (lo que modifica los datos).",
        r"Part of the flux closes outside the domain. Enlarge the field of view to include both polarities, or use \texttt{balance\_flux()} (which alters the data).",
    ),
    "position": (
        "Distancia al meridiano central",
        "Distance from central meridian",
        r"$<30^\circ$ bien; $30$--$45^\circ$ revisar",
        r"$<30^\circ$ good; $30$--$45^\circ$ caution",
        r"Lejos del centro del disco el ruido del campo transversal y los efectos de proyección crecen. Si es posible, elige un instante en que la región esté más cerca del meridiano central.",
        r"Far from disk centre the transverse-field noise and projection effects grow. If possible, choose a time when the region is closer to the central meridian.",
    ),
    "eps_force_raw": (
        r"Fuerza neta (Aly) $\epsilon_F$ en los datos",
        r"Net force (Aly) $\epsilon_F$ in the data",
        r"$<0.1$ bien; $0.1$--$0.3$ revisar",
        r"$<0.1$ good; $0.1$--$0.3$ caution",
        r"Es normal en la fotosfera, que no está libre de fuerzas. Por eso se preprocesa el magnetograma antes del NLFFF.",
        r"This is normal for the photosphere, which is not force-free. That is why the magnetogram is preprocessed before the NLFFF.",
    ),
    "eps_torque_raw": (
        r"Torque neto (Aly) $\epsilon_T$ en los datos",
        r"Net torque (Aly) $\epsilon_T$ in the data",
        r"$<0.1$ bien; $0.1$--$0.3$ revisar",
        r"$<0.1$ good; $0.1$--$0.3$ caution",
        r"Igual que la fuerza neta: lo corrige el preprocesado.",
        r"Same as the net force: preprocessing corrects it.",
    ),
    "eps_force_pp": (
        r"Fuerza neta $\epsilon_F$ tras preprocesar",
        r"Net force $\epsilon_F$ after preprocessing",
        r"$<0.05$ bien; $0.05$--$0.15$ revisar",
        r"$<0.05$ good; $0.05$--$0.15$ caution",
        r"El preprocesado no ha conseguido unos datos compatibles con un campo libre de fuerzas. Prueba a reducir \texttt{mu3} (más libertad para cambiar los datos) o aumentar \texttt{max\_iter}.",
        r"Preprocessing did not make the data compatible with a force-free field. Try a smaller \texttt{mu3} (more freedom to change the data) or a larger \texttt{max\_iter}.",
    ),
    "pp_change": (
        r"Cambio introducido por el preprocesado (rms $|\Delta B|/|B|$)",
        r"Change introduced by preprocessing (rms $|\Delta B|/|B|$)",
        r"$<20\%$ bien; $20$--$40\%$ revisar",
        r"$<20\%$ good; $20$--$40\%$ caution",
        r"El preprocesado está alterando mucho los datos. Aumenta \texttt{mu3} (más peso a los datos) o reduce \texttt{mu4} (menos suavizado).",
        r"Preprocessing is changing the data substantially. Increase \texttt{mu3} (more weight on the data) or decrease \texttt{mu4} (less smoothing).",
    ),
    "alpha_clipped": (
        r"$|\alpha|$ del LFFF respecto a $\alpha_{\max}$",
        r"LFFF $|\alpha|$ relative to $\alpha_{\max}$",
        r"$<90\%$ bien; recortado = problema",
        r"$<90\%$ good; clipped = poor",
        r"La región pide más torsión de la que un LFFF admite en este dominio, o la torsión no es uniforme. El LFFF no la representa bien: trátalo como cota, no como modelo.",
        r"The region requires more twist than an LFFF allows in this domain, or the twist is not uniform. The LFFF cannot represent it: treat it as a bound, not as a model.",
    ),
    "height": (
        "Altura de la caja / lado menor",
        "Box height / shorter side",
        r"$\geq 0.5$ bien; $0.3$--$0.5$ revisar",
        r"$\geq 0.5$ good; $0.3$--$0.5$ caution",
        r"Los bucles grandes se salen por arriba y el techo (fijado al campo potencial) condiciona la solución. Aumenta \texttt{nz}.",
        r"Large loops leave through the top and the upper boundary (fixed to the potential field) constrains the solution. Increase \texttt{nz}.",
    ),
    "converged": (
        "Iteraciones NLFFF / convergencia",
        "NLFFF iterations / convergence",
        r"convergido = bien",
        r"converged = good",
        r"Se alcanzó \texttt{max\_iter} antes de converger. Aumenta \texttt{max\_iter} y comprueba la curva de convergencia.",
        r"\texttt{max\_iter} was reached before convergence. Increase \texttt{max\_iter} and check the convergence curve.",
    ),
    "base_change": (
        r"NLFFF: cambio de la base relajada en campo fuerte (rms $|\Delta B_t|/|B_t|$)",
        r"NLFFF: change of the relaxed base in strong field (rms $|\Delta B_t|/|B_t|$)",
        r"$<15\%$ bien; $15$--$30\%$ revisar",
        r"$<15\%$ good; $15$--$30\%$ caution",
        r"El modelo está modificando mucho el campo transversal donde los datos son fiables. Aumenta \texttt{relax\_bottom} ($\nu$) para pegarlo más a las observaciones.",
        r"The model is changing the transverse field substantially where the data are reliable. Increase \texttt{relax\_bottom} ($\nu$) to keep it closer to the observations.",
    ),
    "e_div": (
        r"NLFFF: energía debida a la divergencia $E_{\rm div}/E$",
        r"NLFFF: energy due to divergence $E_{\rm div}/E$",
        r"$<5\%$ bien; $5$--$10\%$ revisar",
        r"$<5\%$ good; $5$--$10\%$ caution",
        r"Una parte apreciable de la energía se debe a errores de divergencia: la energía libre y sobre todo la helicidad son menos fiables. Más iteraciones o más resolución ayudan.",
        r"An appreciable part of the energy is due to divergence errors: the free energy and above all the helicity are less reliable. More iterations or resolution help.",
    ),
    "gauge": (
        r"NLFFF: diferencia de $H_V$ entre dos gauges",
        r"NLFFF: difference in $H_V$ between two gauges",
        r"$<5\%$ bien; $5$--$15\%$ revisar",
        r"$<5\%$ good; $5$--$15\%$ caution",
        r"La helicidad relativa debe ser independiente del gauge; la diferencia mide su error numérico. Si es grande, da la helicidad sólo como orden de magnitud.",
        r"Relative helicity must be gauge-independent; the difference measures its numerical error. If large, quote the helicity only as an order of magnitude.",
    ),
    "cwsin": (
        r"NLFFF: $\mathrm{CW}_{\sin}$ (ángulo J--B)",
        r"NLFFF: $\mathrm{CW}_{\sin}$ (J--B angle)",
        r"$<0.1$ bien; $0.1$--$0.3$ revisar",
        r"$<0.1$ good; $0.1$--$0.3$ caution",
        r"Quedan fuerzas de Lorentz apreciables. Con datos reales $0.1$--$0.3$ es habitual; valores mayores indican datos inconsistentes o falta de convergencia.",
        r"Significant Lorentz forces remain. With real data $0.1$--$0.3$ is common; larger values indicate inconsistent data or lack of convergence.",
    ),
    "fi": (
        r"NLFFF: divergencia $\langle|f_i|\rangle$",
        r"NLFFF: divergence $\langle|f_i|\rangle$",
        r"$<10^{-3}$ bien; $10^{-3}$--$10^{-2}$ revisar",
        r"$<10^{-3}$ good; $10^{-3}$--$10^{-2}$ caution",
        r"El campo no es del todo solenoidal; la energía puede estar sesgada (Valori et al. 2013). Más iteraciones o mayor resolución ayudan.",
        r"The field is not fully solenoidal; the energy may be biased (Valori et al. 2013). More iterations or higher resolution help.",
    ),
    "e_ratio": (
        r"NLFFF: $E/E_{\rm pot}$",
        r"NLFFF: $E/E_{\rm pot}$",
        r"$1$--$2$ bien; $<1$ problema",
        r"$1$--$2$ good; $<1$ poor",
        r"Un campo libre de fuerzas con el mismo $B_z$ no puede tener menos energía que el potencial (teorema de Thomson). Si sale $<1$, la solución no es fiable (divergencia, contornos inconsistentes).",
        r"A force-free field with the same $B_z$ cannot have less energy than the potential one (Thomson's theorem). A value $<1$ means the solution is unreliable (divergence, inconsistent boundaries).",
    ),
    "open_lines": (
        "NLFFF: líneas que salen del dominio",
        "NLFFF: field lines leaving the domain",
        r"$<50\%$ bien; $50$--$75\%$ revisar",
        r"$<50\%$ good; $50$--$75\%$ caution",
        r"El dominio es pequeño para la estructura magnética: mucha conectividad depende de contornos no observados. Amplía el campo de visión o la altura.",
        r"The domain is small for the magnetic structure: much of the connectivity depends on unobserved boundaries. Enlarge the field of view or the height.",
    ),
    "alpha_const": (
        r"NLFFF: variación relativa de $\alpha$ a lo largo de las líneas",
        r"NLFFF: relative variation of $\alpha$ along field lines",
        r"$<0.3$ bien; $0.3$--$0.6$ revisar",
        r"$<0.3$ good; $0.3$--$0.6$ caution",
        r"En un campo libre de fuerzas $\alpha$ es constante en cada línea. Una variación grande indica que la solución no es del todo libre de fuerzas o que la resolución es insuficiente.",
        r"In a force-free field $\alpha$ is constant along each line. Large variation means the solution is not fully force-free or the resolution is insufficient.",
    ),
    "cvec_exact": (
        r"NLFFF frente al exacto: $C_{\rm vec}$",
        r"NLFFF versus exact: $C_{\rm vec}$",
        r"$\geq 0.99$ bien; $0.95$--$0.99$ revisar",
        r"$\geq 0.99$ good; $0.95$--$0.99$ caution",
        r"El NLFFF no reproduce del todo la solución exacta. Aumenta la resolución o \texttt{max\_iter}.",
        r"The NLFFF does not fully reproduce the exact solution. Increase resolution or \texttt{max\_iter}.",
    ),
    "alpha_sign": (
        r"Signo de $\alpha$: ajuste frente a $J_z/B_z$",
        r"Sign of $\alpha$: fit versus $J_z/B_z$",
        "mismo signo = bien",
        "same sign = good",
        r"Los dos estimadores de $\alpha$ discrepan en signo: la región probablemente tiene helicidad mixta, y un único $\alpha$ no la describe.",
        r"The two $\alpha$ estimators disagree in sign: the region probably has mixed helicity, which a single $\alpha$ cannot describe.",
    ),
    "no_vector": (
        "Magnetograma sin componentes transversales",
        "Magnetogram without transverse components",
        "", "",
        r"Sin $B_x, B_y$ no se puede calcular el NLFFF ni estimar $\alpha$. Usa los segmentos Bp y Bt de SHARP.",
        r"Without $B_x, B_y$ the NLFFF cannot be computed and $\alpha$ cannot be estimated. Use the SHARP Bp and Bt segments.",
    ),
}

GLOSSARY = [
    (r"$\alpha$ (parámetro libre de fuerzas)", r"$\alpha$ (force-free parameter)",
     r"Cociente entre la corriente paralela y el campo: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$. Mide cuánto está retorcido el campo (en Mm$^{-1}$). Signo positivo: torsión dextrógira (helicidad positiva).",
     r"Ratio of the parallel current to the field: $\nabla\times\mathbf{B}=\alpha\mathbf{B}$. Measures how twisted the field is (in Mm$^{-1}$). Positive sign: right-handed twist (positive helicity)."),
    (r"Ambigüedad de $180^\circ$", r"$180^\circ$ ambiguity",
     r"El efecto Zeeman da la dirección del campo transversal pero no su sentido. HMI la resuelve con el método de mínima energía antes de publicar los SHARP.",
     r"The Zeeman effect gives the orientation of the transverse field but not its sense. HMI resolves it with the minimum-energy method before releasing SHARPs."),
    (r"$\beta$ del plasma", r"Plasma $\beta$",
     r"Cociente entre presión del gas y presión magnética. En la corona baja $\beta\ll1$: el campo domina y debe estar casi libre de fuerzas.",
     r"Ratio of gas to magnetic pressure. In the low corona $\beta\ll1$: the field dominates and must be nearly force-free."),
    ("CEA", "CEA",
     r"Proyección cilíndrica de igual área. Los SHARP CEA reproyectan la región a una malla plana, como si se viera desde encima.",
     r"Cylindrical equal-area projection. SHARP CEA data remap the region onto a flat grid, as if seen from directly above."),
    ("Conectividad", "Connectivity",
     r"Qué punto de la fotosfera está unido con cuál a través de una línea de campo. Determina dónde se depositará la energía en una fulguración.",
     r"Which photospheric point is linked to which by a field line. It determines where energy is deposited in a flare."),
    (r"Corriente $\mathbf{J}$", r"Current $\mathbf{J}$",
     r"$\mathbf{J}=\nabla\times\mathbf{B}/\mu_0$. Las corrientes almacenan la energía libre; un campo potencial no tiene corrientes.",
     r"$\mathbf{J}=\nabla\times\mathbf{B}/\mu_0$. Currents store the free energy; a potential field has none."),
    ("Energía libre", "Free energy",
     r"$E_{\rm free}=E-E_{\rm pot}$: energía por encima del mínimo posible para ese $B_z$. Es la cota superior de la energía que pueden liberar fulguraciones y CMEs.",
     r"$E_{\rm free}=E-E_{\rm pot}$: energy above the minimum possible for that $B_z$. It is the upper bound on the energy that flares and CMEs can release."),
    ("Fuerza de Lorentz", "Lorentz force",
     r"$\mathbf{J}\times\mathbf{B}$: fuerza que ejerce el campo sobre el plasma. Es nula en un campo libre de fuerzas.",
     r"$\mathbf{J}\times\mathbf{B}$: force exerted by the field on the plasma. It vanishes in a force-free field."),
    ("HARP / SHARP", "HARP / SHARP",
     r"HMI Active Region Patch: recorte automático de cada región activa. SHARP son los productos derivados (campo vectorial y parámetros).",
     r"HMI Active Region Patch: automatic cut-out of each active region. SHARPs are the derived products (vector field and parameters)."),
    ("Helicidad", "Helicity",
     r"Medida de la torsión y el entrelazado del campo. Regla hemisférica: predomina la helicidad negativa en el norte y la positiva en el sur.",
     r"Measure of the twist and linkage of the field. Hemispheric rule: negative helicity predominates in the north, positive in the south."),
    ("Línea de inversión de polaridad (PIL)", "Polarity inversion line (PIL)",
     r"Curva en la fotosfera donde $B_z$ cambia de signo. La cizalla y las corrientes suelen concentrarse aquí; es donde se forman filamentos y fulguraciones.",
     r"Curve on the photosphere where $B_z$ changes sign. Shear and currents usually concentrate here; it is where filaments and flares form."),
    ("Magnetograma vectorial", "Vector magnetogram",
     r"Mapa de las tres componentes del campo en la fotosfera, obtenido invirtiendo los parámetros de Stokes de la luz polarizada.",
     r"Map of the three field components at the photosphere, obtained by inverting the Stokes parameters of polarised light."),
    ("Preprocesado", "Preprocessing",
     r"Modificación suave del magnetograma para que cumpla las condiciones de un campo libre de fuerzas (fuerza y torque netos nulos) antes del NLFFF.",
     r"Gentle modification of the magnetogram so that it satisfies the conditions of a force-free field (zero net force and torque) before the NLFFF."),
    ("Semilla", "Seed point",
     r"Punto desde el que se integra una línea de campo. Usar las mismas semillas en todos los modelos permite comparar su conectividad.",
     r"Point from which a field line is integrated. Using the same seeds in every model allows their connectivity to be compared."),
    ("Helicidad relativa", "Relative helicity",
     r"$H_V=\int(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV$: helicidad del campo respecto al potencial con el mismo flujo en las caras. Mide la torsión y el entrelazado global (en Mx$^2$).",
     r"$H_V=\int(\mathbf{A}+\mathbf{A}_p)\cdot(\mathbf{B}-\mathbf{B}_p)\,dV$: helicity of the field relative to the potential field with the same boundary flux. Measures global twist and linkage (in Mx$^2$)."),
    ("Índice de eruptividad $|H_J|/|H_V|$", "Eruptivity index $|H_J|/|H_V|$",
     r"Fracción de la helicidad que reside en la parte con corrientes del campo; propuesto como indicador de la capacidad de una región para entrar en erupción (Pariat et al. 2017).",
     r"Fraction of the helicity residing in the current-carrying part of the field; proposed as an indicator of a region's ability to erupt (Pariat et al. 2017)."),
    ("AUC", "AUC",
     r"Área bajo la curva ROC: probabilidad de que un clasificador puntúe peor a un caso malo que a uno bueno. 0.5 = azar; 1 = perfecto.",
     r"Area under the ROC curve: probability that a classifier scores a bad case worse than a good one. 0.5 = chance; 1 = perfect."),
    ("Confianza de una línea", "Field-line confidence",
     r"Probabilidad estimada (0--1) de que una línea del modelo esté bien, combinando indicadores calibrados con una solución exacta.",
     r"Estimated probability (0--1) that a model field line is correct, combining indicators calibrated against an exact solution."),
    ("Cuerda de flujo", "Flux rope",
     r"Haz de líneas de campo enrolladas unas alrededor de otras ($|T_w|\gtrsim1$). Es la estructura que suele entrar en erupción en las eyecciones de masa coronal.",
     r"Bundle of field lines wound around each other ($|T_w|\gtrsim1$). It is the structure that usually erupts in coronal mass ejections."),
    ("Índice de decaimiento $n$", "Decay index $n$",
     r"$n=-d\ln B_h/d\ln z$: rapidez con que decae con la altura el campo externo que sujeta una cuerda de flujo.",
     r"$n=-d\ln B_h/d\ln z$: how fast the external field holding a flux rope down decays with height."),
    ("Inestabilidad de toro", "Torus instability",
     r"Una cuerda de flujo se escapa si el campo externo decae con la altura más deprisa que $n\approx1.5$. La altura a la que ocurre es $h_{\rm crit}$.",
     r"A flux rope escapes if the external field decays with height faster than $n\approx1.5$. The height where this happens is $h_{\rm crit}$."),
    ("Inestabilidad kink", "Kink instability",
     r"Una cuerda de flujo demasiado torsionada ($\sim$1.25--1.75 vueltas) se retuerce sobre sí misma, como un cable de teléfono.",
     r"A flux rope that is too twisted ($\sim$1.25--1.75 turns) writhes upon itself, like a telephone cord."),
    ("QSL y factor $Q$", "QSL and the $Q$ factor",
     r"Capas cuasi-separatrices: zonas donde la conectividad cambia bruscamente ($Q\gg2$). Concentran corrientes y favorecen la reconexión; sus huellas coinciden con las cintas de las fulguraciones.",
     r"Quasi-separatrix layers: regions where connectivity changes abruptly ($Q\gg2$). They concentrate currents and favour reconnection; their footprints match flare ribbons."),
    ("Twist $T_w$", "Twist $T_w$",
     r"$T_w=(4\pi)^{-1}\int\alpha\,dl$: número de vueltas que dan las líneas vecinas alrededor de una línea dada.",
     r"$T_w=(4\pi)^{-1}\int\alpha\,dl$: number of turns neighbouring field lines make around a given line."),
]

REFERENCES = [
    ("Aly1989", "Aly, J. J. 1989, Sol. Phys., 120, 19"),
    ("Alissandrakis1981", "Alissandrakis, C. E. 1981, A\\&A, 100, 197"),
    ("Bobra2014", "Bobra, M. G., Sun, X., Hoeksema, J. T., et al. 2014, Sol. Phys., 289, 3549"),
    ("DeRosa2009", "DeRosa, M. L., Schrijver, C. J., Barnes, G., et al. 2009, ApJ, 696, 1780"),
    ("Gary2001", "Gary, G. A. 2001, Sol. Phys., 203, 71"),
    ("Hagino2004", "Hagino, M., \\& Sakurai, T. 2004, PASJ, 56, 831"),
    ("Hoeksema2014", "Hoeksema, J. T., Liu, Y., Hayashi, K., et al. 2014, Sol. Phys., 289, 3483"),
    ("LowLou1990", "Low, B. C., \\& Lou, Y. Q. 1990, ApJ, 352, 343"),
    ("Metcalf2008", "Metcalf, T. R., DeRosa, M. L., Schrijver, C. J., et al. 2008, Sol. Phys., 247, 269"),
    ("Nakagawa1972", "Nakagawa, Y., \\& Raadu, M. A. 1972, Sol. Phys., 25, 127"),
    ("Pevtsov1995", "Pevtsov, A. A., Canfield, R. C., \\& Metcalf, T. R. 1995, ApJ, 440, L109"),
    ("Schou2012", "Schou, J., Scherrer, P. H., Bush, R. I., et al. 2012, Sol. Phys., 275, 229"),
    ("Schrijver2006", "Schrijver, C. J., DeRosa, M. L., Metcalf, T. R., et al. 2006, Sol. Phys., 235, 161"),
    ("Valori2013", "Valori, G., D\\'emoulin, P., Pariat, E., \\& Masson, S. 2013, A\\&A, 553, A38"),
    ("Wheatland2000", "Wheatland, M. S., Sturrock, P. A., \\& Roumeliotis, G. 2000, ApJ, 540, 1150"),
    ("Wiegelmann2004", "Wiegelmann, T. 2004, Sol. Phys., 219, 87"),
    ("Wiegelmann2006", "Wiegelmann, T., Inhester, B., \\& Sakurai, T. 2006a, Sol. Phys., 233, 215"),
    ("Wiegelmann2021", "Wiegelmann, T., \\& Sakurai, T. 2021, Living Rev. Sol. Phys., 18, 1"),
    ('Berger2006', 'Berger, M. A., \\& Prior, C. 2006, J. Phys. A: Math. Gen., 39, 8321'),
    ('Demoulin1996', "D\\'emoulin, P., Henoux, J. C., Priest, E. R., \\& Mandrini, C. H. 1996, A\\&A, 308, 643"),
    ('GoldHoyle1960', 'Gold, T., \\& Hoyle, F. 1960, MNRAS, 120, 89'),
    ('Kliem2006', 'Kliem, B., \\& T\\"or\\"ok, T. 2006, Phys. Rev. Lett., 96, 255002'),
    ('Liu2016', 'Liu, R., Kliem, B., Titov, V. S., et al. 2016, ApJ, 818, 148'),
    ('Pariat2012', "Pariat, E., \\& D\\'emoulin, P. 2012, A\\&A, 541, A78"),
    ('Titov2002', "Titov, V. S., Hornig, G., \\& D\\'emoulin, P. 2002, J. Geophys. Res., 107, 1164"),
    ('Torok2004', 'T\\"or\\"ok, T., Kliem, B., \\& Titov, V. S. 2004, A\\&A, 413, L27'),
    ('Liu1989', 'Liu, D. C., \\& Nocedal, J. 1989, Math. Program., 45, 503'),
    ('Paszke2019', 'Paszke, A., Gross, S., Massa, F., et al. 2019, Advances in Neural Information Processing Systems, 32, 8024'),
    ('Wiegelmann2010', 'Wiegelmann, T., \\& Inhester, B. 2010, A\\&A, 516, A107'),
    ('BergerField1984', 'Berger, M. A., \\& Field, G. B. 1984, J. Fluid Mech., 147, 133'),
    ('DeVore2000', 'DeVore, C. R. 2000, ApJ, 539, 944'),
    ('FinnAntonsen1985', 'Finn, J. M., \\& Antonsen, T. M. 1985, Comments Plasma Phys. Control. Fusion, 9, 111'),
    ('Pariat2017', 'Pariat, E., Leake, J. E., Valori, G., et al. 2017, A\\&A, 601, A125'),
    ('Valori2012', "Valori, G., D\\'emoulin, P., \\& Pariat, E. 2012, Sol. Phys., 278, 347"),
    ('Valori2016', 'Valori, G., Pariat, E., Anfinogentov, S., et al. 2016, Space Sci. Rev., 201, 147'),
    ('Amari2006', 'Amari, T., Boulmezaoud, T. Z., \\& Aly, J. J. 2006, A\\&A, 446, 691'),
    ('GradRubin1958', 'Grad, H., \\& Rubin, H. 1958, Proc. 2nd UN Int. Conf. Peaceful Uses of Atomic Energy, 31, 190'),
    ('Wheatland2007', 'Wheatland, M. S. 2007, Sol. Phys., 245, 251'),
    ('WheatlandRegnier2009', "Wheatland, M. S., \\& R\\'egnier, S. 2009, ApJ, 700, L88"),
    ('Brier1950', 'Brier, G. W. 1950, Mon. Weather Rev., 78, 1'),
    ('Cox1958', 'Cox, D. R. 1958, J. R. Stat. Soc. B, 20, 215'),
    ('Efron1979', 'Efron, B. 1979, Ann. Stat., 7, 1'),
    ('Hanley1982', 'Hanley, J. A., \\& McNeil, B. J. 1982, Radiology, 143, 29'),
    ('HastieTibshirani1986', 'Hastie, T., \\& Tibshirani, R. 1986, Stat. Sci., 1, 297'),
    ('Friedman2001', 'Friedman, J. H. 2001, Ann. Stat., 29, 1189'),
    ('Breiman2001', 'Breiman, L. 2001, Mach. Learn., 45, 5'),
    ('ZadroznyElkan2002', 'Zadrozny, B., \\& Elkan, C. 2002, Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 694'),
    ('NiculescuMizil2005', 'Niculescu-Mizil, A., \\& Caruana, R. 2005, Proc. 22nd Int. Conf. Machine Learning, 625'),
    ('Guo2017', 'Guo, C., Pleiss, G., Sun, Y., \\& Weinberger, K. Q. 2017, Proc. 34th Int. Conf. Machine Learning, PMLR 70, 1321'),
    ('Pedregosa2011', 'Pedregosa, F., Varoquaux, G., Gramfort, A., et al. 2011, J. Mach. Learn. Res., 12, 2825'),
    ('WheatlandLeka2011', 'Wheatland, M. S., \\& Leka, K. D. 2011, ApJ, 728, 112'),
    ('Gilchrist2012', 'Gilchrist, S. A., Wheatland, M. S., \\& Leka, K. D. 2012, Sol. Phys., 276, 133'),
    ('Wiegelmann2012', 'Wiegelmann, T., Thalmann, J. K., Inhester, B., et al. 2012, Sol. Phys., 281, 37'),
    ('DeRosa2015', 'DeRosa, M. L., Wheatland, M. S., Leka, K. D., et al. 2015, ApJ, 811, 107'),
    ('Fleishman2017', 'Fleishman, G. D., Anfinogentov, S., Loukitcheva, M., et al. 2017, ApJ, 839, 30'),
    ('Warren2018', 'Warren, H. P., Crump, N. A., Ugarte-Urra, I., et al. 2018, ApJ, 860, 46'),
    ('Thalmann2019', 'Thalmann, J. K., Linan, L., Pariat, E., \\& Valori, G. 2019, ApJ, 880, L6'),
    ('Thalmann2020', 'Thalmann, J. K., Sun, X., Moraitis, K., \\& Gupta, M. 2020, A\\&A, 643, A153'),
    ('Jarolim2023', 'Jarolim, R., Thalmann, J. K., Veronig, A. M., \\& Podladchikova, T. 2023, Nat. Astron., 7, 1171'),
    ('TitovDemoulin1999', "Titov, V. S., \\& D\\'emoulin, P. 1999, A\\&A, 351, 707"),
    ('WiegelmannTD2006', 'Wiegelmann, T., Inhester, B., Kliem, B., Valori, G., \\& Neukirch, T. 2006b, A\\&A, 453, 737'),
    ('Valori2010', 'Valori, G., Kliem, B., T\\"or\\"ok, T., \\& Titov, V. S. 2010, A\\&A, 519, A44'),
]

BIBTEX = r"""@ARTICLE{Aly1989, author = {{Aly}, J.~J.}, title = "{On the reconstruction of the nonlinear force-free coronal magnetic field from boundary data}", journal = {Sol. Phys.}, year = 1989, volume = {120}, pages = {19-48}, doi = {10.1007/BF00148533}}
@ARTICLE{Alissandrakis1981, author = {{Alissandrakis}, C.~E.}, title = "{On the computation of constant alpha force-free magnetic field}", journal = {A\&A}, year = 1981, volume = {100}, pages = {197-200}}
@ARTICLE{Bobra2014, author = {{Bobra}, M.~G. and {Sun}, X. and {Hoeksema}, J.~T. and others}, title = "{The Helioseismic and Magnetic Imager (HMI) Vector Magnetic Field Pipeline: SHARPs - Space-Weather HMI Active Region Patches}", journal = {Sol. Phys.}, year = 2014, volume = {289}, pages = {3549-3578}, doi = {10.1007/s11207-014-0529-3}}
@ARTICLE{DeRosa2009, author = {{DeRosa}, M.~L. and {Schrijver}, C.~J. and {Barnes}, G. and others}, title = "{A Critical Assessment of Nonlinear Force-Free Field Modeling of the Solar Corona for Active Region 10953}", journal = {ApJ}, year = 2009, volume = {696}, pages = {1780-1791}, doi = {10.1088/0004-637X/696/2/1780}}
@ARTICLE{Gary2001, author = {{Gary}, G.~A.}, title = "{Plasma Beta above a Solar Active Region: Rethinking the Paradigm}", journal = {Sol. Phys.}, year = 2001, volume = {203}, pages = {71-86}, doi = {10.1023/A:1012722021820}}
@ARTICLE{Hagino2004, author = {{Hagino}, M. and {Sakurai}, T.}, title = "{Latitude Variation of Helicity in Solar Active Regions}", journal = {PASJ}, year = 2004, volume = {56}, pages = {831-843}, doi = {10.1093/pasj/56.5.831}}
@ARTICLE{Hoeksema2014, author = {{Hoeksema}, J.~T. and {Liu}, Y. and {Hayashi}, K. and others}, title = "{The Helioseismic and Magnetic Imager (HMI) Vector Magnetic Field Pipeline: Overview and Performance}", journal = {Sol. Phys.}, year = 2014, volume = {289}, pages = {3483-3530}, doi = {10.1007/s11207-014-0516-8}}
@ARTICLE{LowLou1990, author = {{Low}, B.~C. and {Lou}, Y.~Q.}, title = "{Modeling Solar Force-free Magnetic Fields}", journal = {ApJ}, year = 1990, volume = {352}, pages = {343}, doi = {10.1086/168541}}
@ARTICLE{Metcalf2008, author = {{Metcalf}, T.~R. and {DeRosa}, M.~L. and {Schrijver}, C.~J. and others}, title = "{Nonlinear Force-Free Modeling of Coronal Magnetic Fields. II. Modeling a Filament Arcade and Simulated Chromospheric and Photospheric Vector Fields}", journal = {Sol. Phys.}, year = 2008, volume = {247}, pages = {269-299}, doi = {10.1007/s11207-007-9110-7}}
@ARTICLE{Nakagawa1972, author = {{Nakagawa}, Y. and {Raadu}, M.~A.}, title = "{On Practical Representation of Magnetic Field}", journal = {Sol. Phys.}, year = 1972, volume = {25}, pages = {127-135}, doi = {10.1007/BF00155751}}
@ARTICLE{Pevtsov1995, author = {{Pevtsov}, A.~A. and {Canfield}, R.~C. and {Metcalf}, T.~R.}, title = "{Latitudinal Variation of Helicity of Photospheric Magnetic Fields}", journal = {ApJL}, year = 1995, volume = {440}, pages = {L109}, doi = {10.1086/187773}}
@ARTICLE{Schou2012, author = {{Schou}, J. and {Scherrer}, P.~H. and {Bush}, R.~I. and others}, title = "{Design and Ground Calibration of the Helioseismic and Magnetic Imager (HMI) Instrument on the Solar Dynamics Observatory (SDO)}", journal = {Sol. Phys.}, year = 2012, volume = {275}, pages = {229-259}, doi = {10.1007/s11207-011-9842-2}}
@ARTICLE{Schrijver2006, author = {{Schrijver}, C.~J. and {DeRosa}, M.~L. and {Metcalf}, T.~R. and others}, title = "{Nonlinear Force-Free Modeling of Coronal Magnetic Fields Part I: A Quantitative Comparison of Methods}", journal = {Sol. Phys.}, year = 2006, volume = {235}, pages = {161-190}, doi = {10.1007/s11207-006-0068-7}}
@ARTICLE{Valori2013, author = {{Valori}, G. and {D{\'e}moulin}, P. and {Pariat}, E. and {Masson}, S.}, title = "{Accuracy of magnetic energy computations}", journal = {A\&A}, year = 2013, volume = {553}, pages = {A38}, doi = {10.1051/0004-6361/201220982}}
@ARTICLE{Wheatland2000, author = {{Wheatland}, M.~S. and {Sturrock}, P.~A. and {Roumeliotis}, G.}, title = "{An Optimization Approach to Reconstructing Force-free Fields}", journal = {ApJ}, year = 2000, volume = {540}, pages = {1150-1155}, doi = {10.1086/309355}}
@ARTICLE{Wiegelmann2004, author = {{Wiegelmann}, T.}, title = "{Optimization code with weighting function for the reconstruction of coronal magnetic fields}", journal = {Sol. Phys.}, year = 2004, volume = {219}, pages = {87-108}, doi = {10.1023/B:SOLA.0000021799.39465.36}}
@ARTICLE{Wiegelmann2006, author = {{Wiegelmann}, T. and {Inhester}, B. and {Sakurai}, T.}, title = "{Preprocessing of Vector Magnetograph Data for a Nonlinear Force-Free Magnetic Field Reconstruction}", journal = {Sol. Phys.}, year = 2006, volume = {233}, pages = {215-232}, doi = {10.1007/s11207-006-2092-z}}
@ARTICLE{Wiegelmann2021, author = {{Wiegelmann}, T. and {Sakurai}, T.}, title = "{Solar force-free magnetic fields}", journal = {Living Rev. Sol. Phys.}, year = 2021, volume = {18}, pages = {1}, doi = {10.1007/s41116-020-00027-4}}
@ARTICLE{Berger2006, author = {{Berger}, M.~A. and {Prior}, C.}, title = "{The writhe of open and closed curves}", journal = {J. Phys. A: Math. Gen.}, year = 2006, volume = {39}, pages = {8321-8348}, doi = {10.1088/0305-4470/39/26/005}}
@ARTICLE{Demoulin1996, author = {{D{\'e}moulin}, P. and {Henoux}, J.~C. and {Priest}, E.~R. and {Mandrini}, C.~H.}, title = "{Quasi-Separatrix layers in solar flares. I. Method}", journal = {A\&A}, year = 1996, volume = {308}, pages = {643-655}}
@ARTICLE{GoldHoyle1960, author = {{Gold}, T. and {Hoyle}, F.}, title = "{On the origin of solar flares}", journal = {MNRAS}, year = 1960, volume = {120}, pages = {89}, doi = {10.1093/mnras/120.2.89}}
@ARTICLE{Kliem2006, author = {{Kliem}, B. and {T{\"o}r{\"o}k}, T.}, title = "{Torus Instability}", journal = {Phys. Rev. Lett.}, year = 2006, volume = {96}, pages = {255002}, doi = {10.1103/PhysRevLett.96.255002}}
@ARTICLE{Liu2016, author = {{Liu}, R. and {Kliem}, B. and {Titov}, V.~S. and others}, title = "{Structure, Stability, and Evolution of Magnetic Flux Ropes from the Perspective of Magnetic Twist}", journal = {ApJ}, year = 2016, volume = {818}, pages = {148}, doi = {10.3847/0004-637X/818/2/148}}
@ARTICLE{Pariat2012, author = {{Pariat}, E. and {D{\'e}moulin}, P.}, title = "{Estimation of the squashing degree within a three-dimensional domain}", journal = {A\&A}, year = 2012, volume = {541}, pages = {A78}, doi = {10.1051/0004-6361/201118515}}
@ARTICLE{Titov2002, author = {{Titov}, V.~S. and {Hornig}, G. and {D{\'e}moulin}, P.}, title = "{Theory of magnetic connectivity in the solar corona}", journal = {J. Geophys. Res.}, year = 2002, volume = {107}, pages = {1164}, doi = {10.1029/2001JA000278}}
@ARTICLE{Torok2004, author = {{T{\"o}r{\"o}k}, T. and {Kliem}, B. and {Titov}, V.~S.}, title = "{Ideal kink instability of a magnetic loop equilibrium}", journal = {A\&A}, year = 2004, volume = {413}, pages = {L27-L30}, doi = {10.1051/0004-6361:20031691}}
@ARTICLE{Liu1989, author = {{Liu}, D.~C. and {Nocedal}, J.}, title = "{On the limited memory BFGS method for large scale optimization}", journal = {Math. Program.}, year = 1989, volume = {45}, pages = {503-528}, doi = {10.1007/BF01589116}}
@INPROCEEDINGS{Paszke2019, author = {{Paszke}, A. and {Gross}, S. and {Massa}, F. and others}, title = "{PyTorch: An Imperative Style, High-Performance Deep Learning Library}", booktitle = {Advances in Neural Information Processing Systems}, year = 2019, volume = {32}, pages = {8024-8035}}
@ARTICLE{Wiegelmann2010, author = {{Wiegelmann}, T. and {Inhester}, B.}, title = "{How to deal with measurement errors and lacking data in nonlinear force-free coronal magnetic field modelling?}", journal = {A\&A}, year = 2010, volume = {516}, pages = {A107}, doi = {10.1051/0004-6361/201014391}}
@ARTICLE{BergerField1984, author = {{Berger}, M.~A. and {Field}, G.~B.}, title = "{The topological properties of magnetic helicity}", journal = {J. Fluid Mech.}, year = 1984, volume = {147}, pages = {133-148}, doi = {10.1017/S0022112084002019}}
@ARTICLE{DeVore2000, author = {{DeVore}, C.~R.}, title = "{Magnetic Helicity Generation by Solar Differential Rotation}", journal = {ApJ}, year = 2000, volume = {539}, pages = {944-953}, doi = {10.1086/309274}}
@ARTICLE{FinnAntonsen1985, author = {{Finn}, J.~M. and {Antonsen}, T.~M.}, title = "{Magnetic helicity: what is it and what is it good for?}", journal = {Comments Plasma Phys. Control. Fusion}, year = 1985, volume = {9}, pages = {111}}
@ARTICLE{Pariat2017, author = {{Pariat}, E. and {Leake}, J.~E. and {Valori}, G. and others}, title = "{Relative magnetic helicity as a diagnostic of solar eruptivity}", journal = {A\&A}, year = 2017, volume = {601}, pages = {A125}, doi = {10.1051/0004-6361/201630043}}
@ARTICLE{Valori2012, author = {{Valori}, G. and {D{\'e}moulin}, P. and {Pariat}, E.}, title = "{Comparing Values of the Relative Magnetic Helicity in Finite Volumes}", journal = {Sol. Phys.}, year = 2012, volume = {278}, pages = {347-366}, doi = {10.1007/s11207-012-9951-6}}
@ARTICLE{Valori2016, author = {{Valori}, G. and {Pariat}, E. and {Anfinogentov}, S. and others}, title = "{Magnetic Helicity Estimations in Models and Observations of the Solar Magnetic Field. Part I: Finite Volume Methods}", journal = {Space Sci. Rev.}, year = 2016, volume = {201}, pages = {147-200}, doi = {10.1007/s11214-016-0299-3}}
@ARTICLE{Amari2006, author = {{Amari}, T. and {Boulmezaoud}, T.~Z. and {Aly}, J.~J.}, title = "{Well posed reconstruction of the solar coronal magnetic field}", journal = {A\&A}, year = 2006, volume = {446}, pages = {691-705}, doi = {10.1051/0004-6361:20054076}}
@INPROCEEDINGS{GradRubin1958, author = {{Grad}, H. and {Rubin}, H.}, title = "{Hydromagnetic Equilibria and Force-Free Fields}", booktitle = {Proc. 2nd UN Int. Conf. on the Peaceful Uses of Atomic Energy}, year = 1958, volume = {31}, pages = {190}}
@ARTICLE{Wheatland2007, author = {{Wheatland}, M.~S.}, title = "{Calculating and Testing Nonlinear Force-Free Fields}", journal = {Sol. Phys.}, year = 2007, volume = {245}, pages = {251-262}, doi = {10.1007/s11207-007-9054-y}}
@ARTICLE{WheatlandRegnier2009, author = {{Wheatland}, M.~S. and {R{\'e}gnier}, S.}, title = "{A Self-Consistent Nonlinear Force-Free Solution for a Solar Active Region Magnetic Field}", journal = {ApJ}, year = 2009, volume = {700}, pages = {L88-L91}, doi = {10.1088/0004-637X/700/2/L88}}
@ARTICLE{Brier1950, author = {{Brier}, G.~W.}, title = "{Verification of forecasts expressed in terms of probability}", journal = {Mon. Weather Rev.}, year = 1950, volume = {78}, pages = {1-3}, doi = {10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2}}
@ARTICLE{Cox1958, author = {{Cox}, D.~R.}, title = "{The regression analysis of binary sequences}", journal = {J. R. Stat. Soc. B}, year = 1958, volume = {20}, pages = {215-242}, doi = {10.1111/j.2517-6161.1958.tb00292.x}}
@ARTICLE{Efron1979, author = {{Efron}, B.}, title = "{Bootstrap methods: another look at the jackknife}", journal = {Ann. Stat.}, year = 1979, volume = {7}, pages = {1-26}, doi = {10.1214/aos/1176344552}}
@ARTICLE{Hanley1982, author = {{Hanley}, J.~A. and {McNeil}, B.~J.}, title = "{The meaning and use of the area under a receiver operating characteristic (ROC) curve}", journal = {Radiology}, year = 1982, volume = {143}, pages = {29-36}, doi = {10.1148/radiology.143.1.7063747}}
@ARTICLE{HastieTibshirani1986, author = {{Hastie}, T. and {Tibshirani}, R.}, title = "{Generalized additive models}", journal = {Stat. Sci.}, year = 1986, volume = {1}, pages = {297-310}, doi = {10.1214/ss/1177013604}}
@ARTICLE{Friedman2001, author = {{Friedman}, J.~H.}, title = "{Greedy function approximation: a gradient boosting machine}", journal = {Ann. Stat.}, year = 2001, volume = {29}, pages = {1189-1232}, doi = {10.1214/aos/1013203451}}
@ARTICLE{Breiman2001, author = {{Breiman}, L.}, title = "{Random forests}", journal = {Mach. Learn.}, year = 2001, volume = {45}, pages = {5-32}, doi = {10.1023/A:1010933404324}}
@INPROCEEDINGS{ZadroznyElkan2002, author = {{Zadrozny}, B. and {Elkan}, C.}, title = "{Transforming classifier scores into accurate multiclass probability estimates}", booktitle = {Proc. 8th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining}, year = 2002, pages = {694-699}, doi = {10.1145/775047.775151}}
@INPROCEEDINGS{NiculescuMizil2005, author = {{Niculescu-Mizil}, A. and {Caruana}, R.}, title = "{Predicting good probabilities with supervised learning}", booktitle = {Proc. 22nd Int. Conf. Machine Learning}, year = 2005, pages = {625-632}, doi = {10.1145/1102351.1102430}}
@INPROCEEDINGS{Guo2017, author = {{Guo}, C. and {Pleiss}, G. and {Sun}, Y. and {Weinberger}, K.~Q.}, title = "{On calibration of modern neural networks}", booktitle = {Proc. 34th Int. Conf. Machine Learning}, series = {PMLR}, volume = {70}, year = 2017, pages = {1321-1330}}
@ARTICLE{Pedregosa2011, author = {{Pedregosa}, F. and {Varoquaux}, G. and {Gramfort}, A. and others}, title = "{Scikit-learn: machine learning in Python}", journal = {J. Mach. Learn. Res.}, year = 2011, volume = {12}, pages = {2825-2830}, doi = {10.48550/arXiv.1201.0490}}
@ARTICLE{WheatlandLeka2011, author = {{Wheatland}, M.~S. and {Leka}, K.~D.}, title = "{Achieving self-consistent nonlinear force-free modeling of solar active regions}", journal = {ApJ}, year = 2011, volume = {728}, pages = {112}, doi = {10.1088/0004-637X/728/2/112}}
@ARTICLE{Gilchrist2012, author = {{Gilchrist}, S.~A. and {Wheatland}, M.~S. and {Leka}, K.~D.}, title = "{The free energy of NOAA solar active region AR 11029}", journal = {Sol. Phys.}, year = 2012, volume = {276}, pages = {133-160}, doi = {10.1007/s11207-011-9878-3}}
@ARTICLE{Wiegelmann2012, author = {{Wiegelmann}, T. and {Thalmann}, J.~K. and {Inhester}, B. and {Tadesse}, T. and {Sun}, X. and {Hoeksema}, J.~T.}, title = "{How should one optimize nonlinear force-free coronal magnetic field extrapolations from SDO/HMI vector magnetograms?}", journal = {Sol. Phys.}, year = 2012, volume = {281}, pages = {37-51}, doi = {10.1007/s11207-012-9966-z}}
@ARTICLE{DeRosa2015, author = {{DeRosa}, M.~L. and {Wheatland}, M.~S. and {Leka}, K.~D. and {Barnes}, G. and {Amari}, T. and {Canou}, A. and {Gilchrist}, S.~A. and {Thalmann}, J.~K. and {Valori}, G. and {Wiegelmann}, T. and {Schrijver}, C.~J. and {Malanushenko}, A. and {Sun}, X. and {R{\'e}gnier}, S.}, title = "{The influence of spatial resolution on nonlinear force-free modeling}", journal = {ApJ}, year = 2015, volume = {811}, pages = {107}, doi = {10.1088/0004-637X/811/2/107}}
@ARTICLE{Fleishman2017, author = {{Fleishman}, G.~D. and {Anfinogentov}, S. and {Loukitcheva}, M. and {Mysh'yakov}, I. and {Stupishin}, A.}, title = "{Casting the coronal magnetic field reconstruction tools in 3D using the MHD Bifrost model}", journal = {ApJ}, year = 2017, volume = {839}, pages = {30}, doi = {10.3847/1538-4357/aa6840}}
@ARTICLE{Warren2018, author = {{Warren}, H.~P. and {Crump}, N.~A. and {Ugarte-Urra}, I. and {Sun}, X. and {Aschwanden}, M.~J. and {Wiegelmann}, T.}, title = "{Toward a quantitative comparison of magnetic field extrapolations and observed coronal loops}", journal = {ApJ}, year = 2018, volume = {860}, pages = {46}, doi = {10.3847/1538-4357/aac20b}}
@ARTICLE{Thalmann2019, author = {{Thalmann}, J.~K. and {Linan}, L. and {Pariat}, E. and {Valori}, G.}, title = "{On the reliability of magnetic energy and helicity computations based on nonlinear force-free coronal magnetic field models}", journal = {ApJL}, year = 2019, volume = {880}, pages = {L6}, doi = {10.3847/2041-8213/ab2e73}}
@ARTICLE{Thalmann2020, author = {{Thalmann}, J.~K. and {Sun}, X. and {Moraitis}, K. and {Gupta}, M.}, title = "{Deducing the reliability of relative helicities from nonlinear force-free coronal models}", journal = {A\&A}, year = 2020, volume = {643}, pages = {A153}, doi = {10.1051/0004-6361/202038921}}
@ARTICLE{Jarolim2023, author = {{Jarolim}, R. and {Thalmann}, J.~K. and {Veronig}, A.~M. and {Podladchikova}, T.}, title = "{Probing the solar coronal magnetic field with physics-informed neural networks}", journal = {Nature Astronomy}, year = 2023, volume = {7}, pages = {1171-1179}, doi = {10.1038/s41550-023-02030-9}}
@ARTICLE{TitovDemoulin1999, author = {{Titov}, V.~S. and {D{\'e}moulin}, P.}, title = "{Basic topology of twisted magnetic configurations in solar flares}", journal = {A\&A}, year = 1999, volume = {351}, pages = {707-720}}
@ARTICLE{WiegelmannTD2006, author = {{Wiegelmann}, T. and {Inhester}, B. and {Kliem}, B. and {Valori}, G. and {Neukirch}, T.}, title = "{Testing non-linear force-free coronal magnetic field extrapolations with the Titov-D{\'e}moulin equilibrium}", journal = {A\&A}, year = 2006, volume = {453}, pages = {737-741}, doi = {10.1051/0004-6361:20054751}}
@ARTICLE{Valori2010, author = {{Valori}, G. and {Kliem}, B. and {T{\"o}r{\"o}k}, T. and {Titov}, V.~S.}, title = "{Testing magnetofrictional extrapolation with the Titov-D{\'e}moulin model of solar active regions}", journal = {A\&A}, year = 2010, volume = {519}, pages = {A44}, doi = {10.1051/0004-6361/201014416}}
"""

REFERENCES.sort(key=lambda r: r[1].lower())
GLOSSARY.sort(key=lambda g: g[0].lower().lstrip('$\\'))


def _author_year(key: str) -> tuple[str, str]:
    """('Wiegelmann et al.', '2006') from the REFERENCES entry."""
    ref = dict(REFERENCES)[key]
    m = re.search(r"\b(1[89]\d\d|20\d\d)[a-z]?\b", ref)  # "2006a", "2006b" if two are the same
    authors, year = ref[:m.start()].strip().rstrip(","), m.group(0)
    first = authors.split(",")[0].strip()
    if "et al" in authors or authors.count(r"\&") == 0 and authors.count(",") > 2:
        return f"{first} et al.", year
    if r"\&" in authors:
        parts = [a.strip() for a in authors.split(r"\&")]
        if parts[0].count(",") > 2:  # three or more authors
            return f"{first} et al.", year
        return rf"{first} \& {parts[1].split(',')[0].strip()}", year
    return first, year


def resolve_citations(tex: str) -> str:
    r"""Convert ``\citep[pre][post]{a,b}``, ``\citet{a}`` and ``\citealt{a}`` into author-year text.

    The report does not use BibTeX (the reference list is written directly),
    so the citations are resolved here; ``paper/references.bib`` contains the
    same keys to use ``\citep`` in the paper.
    """
    def rep(m):
        cmd, pre, post, keys = m.group(1), m.group(2), m.group(3), m.group(4)
        if post is None and pre is not None:  # \citep[post]{...}
            pre, post = None, pre
        items = [_author_year(k.strip()) for k in keys.split(",")]
        if cmd == "citet":
            return "; ".join(f"{a} ({y})" for a, y in items)
        if cmd == "citealt":
            return "; ".join(f"{a} {y}" for a, y in items)
        body = "; ".join(f"{a} {y}" for a, y in items)
        if pre:
            body = f"{pre} {body}"
        if post:
            body = f"{body}, {post}"
        return f"({body})"
    return re.sub(r"\\(citep|citet|citealt)(?:\[([^\]]*)\])?(?:\[([^\]]*)\])?\{([^}]*)\}", rep, tex)
