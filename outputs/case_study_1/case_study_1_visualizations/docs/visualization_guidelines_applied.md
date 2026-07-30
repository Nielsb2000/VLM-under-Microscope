# Visualization guidelines applied

The figures were designed around the following manuscript-figure principles:

1. **Lead with the comparison that matches the experimental design.** Because some cases share the same sample seed, randomization seed, and randomized filters, paired plots are preferred over only showing aggregate distributions.
2. **Make the visual question obvious.** Axes explicitly state that lower final score is better, and paired-delta plots state that negative values favor the exploratory prompt.
3. **Use accessible colors.** The method palette uses Okabe-Ito-style colorblind-safe blue and vermillion, plus neutral gray for randomized starts.
4. **Reduce chartjunk.** The plots avoid heavy grids, 3D effects, unnecessary frames, and decorative elements.
5. **Export for papers.** Figures are written as PDF and SVG vector graphics plus 600-DPI PNG fallbacks.
6. **Use consistent typography and sizing.** The style module sets one-column-friendly dimensions, consistent font sizes, embedded TrueType fonts in PDF, and uncluttered legends.
7. **Show individual data points where n is not huge.** Boxplots are overlaid with jittered points; paired plots show each matched case rather than only a mean.
8. **Separate quantitative evidence from qualitative examples.** Contact sheets are generated for illustration, while paired quantitative figures remain the main evidence.
9. **Keep reproducibility visible.** The pipeline writes normalized tidy data, matched-pair data, and summary statistics next to the figures.

Potential final-paper caption for the main matched figure:

> Matched comparison of direct optimization and exploratory prompting for case study 1. Each line connects two runs initialized with the same SEM image, randomized filter settings, sample seed, and randomization seed. Lower histogram-distance scores indicate closer agreement with the hidden reference histogram. Downward lines therefore indicate cases where the exploratory prompt improved optimization outcome relative to direct optimization.
