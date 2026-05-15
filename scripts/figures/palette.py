"""
Neural Haze palette — shared color constants for figure scripts.

Match the same palette used in the analysis notebook so figures are visually
consistent across the project.
"""

# Background / structural
VOID        = '#0A0A14'   # darkest background
DEEP_SPACE  = '#15152A'   # panel background
SLATE       = '#3A3A55'   # axes, gridlines (subtle)
MIST        = '#8B8BA8'   # tick labels, secondary text
GHOST       = '#E8E8F5'   # primary text on dark bg

# Highlight colors
MAGENTA     = '#FF2D95'
CYAN        = '#00D9FF'
VIOLET      = '#7B5CFF'
ACID_LIME   = '#C6FF3D'

# Series-1 default order (use this when plotting multiple series)
CHART_SERIES = [CYAN, VIOLET, ACID_LIME, MAGENTA]

# CV-specific assignments (consistent across charts)
CV_COLORS = {
    'cv_primary':  ACID_LIME,
    'cv_hr':       CYAN,
    'cv_engineer': VIOLET,
}
CV_LABELS = {
    'cv_primary':  'Assessment Scientist',
    'cv_hr':       'Head of People',
    'cv_engineer': 'Solution Architect',
}
# Short variant with "CV" suffix — use in chart legends where space is tight
CV_LABELS_SHORT = {
    'cv_primary':  'Assessment Scientist CV',
    'cv_hr':       'HR CV',
    'cv_engineer': 'Engineering CV',
}
CV_KEYS = ['cv_primary', 'cv_hr', 'cv_engineer']

# JD pool naming convention (per saved feedback memory):
# domain-led name with legacy code name in parens
POOL_KEYS   = ['main', 'hr_extra', 'engineer_extra']
POOL_LABELS = {
    'main':           'Psychometric JDs (main)',
    'hr_extra':       'HR JDs (hr_extra)',
    'engineer_extra': 'Engineering JDs (engineer_extra)',
}

ALL_COLOR = MAGENTA  # used for the pooled "All CVs" line


def style_axes(ax):
    """Apply Neural Haze styling to a matplotlib axis."""
    ax.set_facecolor(VOID)
    ax.tick_params(colors=MIST, length=0)
    for spine in ax.spines.values():
        spine.set_color(SLATE)
        spine.set_alpha(0.5)
    ax.grid(color=SLATE, alpha=0.3, linewidth=0.6, zorder=0)
