"""Design: colors, fonts and shared styling for charts and tables."""

BG, PANEL, GRID, HEADER = '#0b0e14', '#0d1117', '#1c2230', '#161b22'
TXT, MUTED = '#e5e7eb', '#6b7280'
GREEN, RED, YELLOW, GRAY, BLUE, CYAN = '#22c55e', '#ef4444', '#eab308', '#475569', '#3b82f6', '#22d3ee'
FONT = 'Consolas, monospace'

# zone band colors (rgb) and opacity per timeframe
ZONE_RGB = {'sup': ('34,197,94', 'SUP'), 'res': ('234,179,8', 'RES')}
ZONE_ALPHA = {'15m': 0.18, '1h': 0.10, '4h': 0.07, '1d': 0.05}

# EMA line colors (the chart draws the 3 EMAs of the selected test)
EMA_COLORS = {9: '#facc15', 13: '#facc15', 21: '#a855f7', 48: '#a855f7', 200: '#ef4444'}

# entry-plan strip colors
STATE_COLORS = {'BUY': GREEN, 'SELL': RED, 'BLOCKED': YELLOW, 'SKIPPED': CYAN, 'NO ENTRY': GRAY}

# per-cell text colors for the rule table
CELL_COLORS = {'BULL': f'color: {GREEN}', 'BUY': f'color: {GREEN}; font-weight: bold',
               'BEAR': f'color: {RED}', 'SELL': f'color: {RED}; font-weight: bold',
               'NONE': f'color: {MUTED}', '-': f'color: {MUTED}',
               'BLOCKED': f'color: {YELLOW}', 'SKIPPED': f'color: {CYAN}', 'NO ENTRY': f'color: {MUTED}'}


def base_layout(title: str, height: int) -> dict:
    return dict(title=dict(text=title, font=dict(size=13, color=TXT), x=0.01, y=0.985),
                paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TXT, family=FONT),
                height=height, bargap=0.2, hovermode='x unified', margin=dict(l=20, r=80, t=80, b=10),
                legend=dict(orientation='h', y=1.035, x=0, font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
                hoverlabel=dict(bgcolor=HEADER, font=dict(color=TXT)))
