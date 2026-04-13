
# --- Minimal Dash app to display the parsed results DataFrame as an HTML table ---
import os
import sys

import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from json_results_to_df import load_results_df
sys.path.append(os.path.dirname(__file__))

df_results = load_results_df()
app = dash.Dash(__name__)


# Dropdown options
reasoning_options = [
    {"label": "Low", "value": "low"},
    {"label": "Medium", "value": "medium"},
    {"label": "High", "value": "high"},
]
image_type_options = [
    {"label": "Color", "value": "color"},
    {"label": "Greyscale", "value": "greyscale"},
    {"label": "Inverted Greyscale", "value": "inverted-greyscale"},
]
blur_level_options = [
    {"label": "No Blur", "value": "no_blur"},
    {"label": "Medium Blur", "value": "med_blur"},
    {"label": "Heavy Blur", "value": "heavy_blur"},
]

def df_to_dash_table(df):
    if df.empty:
        return html.Div([html.H3("No JSON results found. Please ensure the Results/ directory contains .json evaluation outputs.")])
    # Limit to first 1000 rows for browser performance
    df_disp = df.head(1000)
    # Hide reasoning_mode column if all values are None/empty
    cols = list(df_disp.columns)
    if 'reasoning_mode' in cols and df_disp['reasoning_mode'].isnull().all():
        cols.remove('reasoning_mode')
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in cols])),
        html.Tbody([
            html.Tr([html.Td(df_disp.iloc[i][col]) for col in cols]) for i in range(len(df_disp))
        ])
    ])


# --- Heatmap utilities for all models ---
def compute_heatmap_data(df, model, reasoning_mode=None):
    df_model = df[df['Model'] == model]
    if reasoning_mode is not None and 'reasoning_mode' in df_model.columns:
        df_model = df_model[df_model['reasoning_mode'] == reasoning_mode]
    image_types = ['color', 'greyscale', 'inverted-greyscale']
    blur_levels = ['no_blur', 'med_blur', 'heavy_blur']
    z = []
    for img_type in image_types:
        row = []
        for blur in blur_levels:
            subset = df_model[(df_model['image_type'] == img_type) & (df_model['blur_level'] == blur)]
            if len(subset) == 0:
                acc = None
            else:
                acc = subset['Correct'].mean()
            row.append(acc)
        z.append(row)
    return image_types, blur_levels, z

def model_heatmap_figure(df, model, reasoning_mode=None):
    image_types, blur_levels, z = compute_heatmap_data(df, model, reasoning_mode)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=blur_levels,
        y=image_types,
        colorscale=[
            [0.0, 'red'],
            [0.5, 'yellow'],
            [1.0, 'green']
        ],
        colorbar=dict(title='Accuracy'),
        zmin=0, zmax=1
    ))
    fig.update_layout(
        xaxis_title='Blur Level',
        yaxis_title='Image Type',
        title=f'{model} Accuracy Heatmap',
        height=400,
        width=400
    )
    return dcc.Graph(figure=fig)


def build_stacked_skills_figure(df, category):
    """Aggregate accuracy by model/skills for every level in the chosen category."""
    models = ['gpt-4o', 'gpt-5.1', 'gpt-5.2']
    skills_modes = ['skills', 'no_skills']
    # Assign distinct colors for each model/skills combination
    combo_colors = {
        ('gpt-4o', 'skills'): '#1f77b4',        # blue
        ('gpt-4o', 'no_skills'): '#87ceeb',     # light blue
        ('gpt-5.1', 'skills'): '#2ca02c',       # green
        ('gpt-5.1', 'no_skills'): '#90ee90',    # light green
        ('gpt-5.2', 'skills'): '#d62728',       # red
        ('gpt-5.2', 'no_skills'): '#ff7f0e'     # orange
    }
    level_map = {
        'reasoning_mode': ['low', 'medium', 'high'],
        'blur_level': ['no_blur', 'med_blur', 'heavy_blur'],
        'image_type': ['color', 'greyscale', 'inverted-greyscale']
    }
    levels = level_map.get(category, [])
    if not levels:
        return go.Figure()
    data = {level: {model: {mode: 0 for mode in skills_modes} for model in models} for level in levels}
    for level in levels:
        for model in models:
            for skills_mode in skills_modes:
                df_slice = df[(df['Model'] == model) & (df['skills_mode'] == skills_mode)]
                if category == 'reasoning_mode':
                    if model == 'gpt-4o' and level != 'low':
                        df_slice = df_slice.iloc[0:0]
                    else:
                        df_slice = df_slice[df_slice['reasoning_mode'] == level]
                elif category == 'blur_level':
                    df_slice = df_slice[df_slice['blur_level'] == level]
                elif category == 'image_type':
                    df_slice = df_slice[df_slice['image_type'] == level]
                acc = df_slice['Correct'].mean() if not df_slice.empty else 0
                data[level][model][skills_mode] = acc
    fig = go.Figure()
    # For each level, show grouped bars for models, stacked for skills/no_skills
    for skills_mode in ['no_skills', 'skills']:
        for model in models:
            fig.add_bar(
                x=levels,
                y=[data[level][model][skills_mode] for level in levels],
                name=f"{model} - {skills_mode.replace('_', ' ').title()}",
                marker_color=combo_colors[(model, skills_mode)],
                offsetgroup=model,
                base=None,
                legendgroup=f"{model}-{skills_mode}",
                showlegend=True
            )
    fig.update_layout(
        barmode='overlay',
        yaxis=dict(title='Accuracy', range=[0, 1]),
        xaxis=dict(title=category.replace('_', ' ').title()),
        title='Effectiveness of Skills vs No Skills',
        height=600,
        width=800,
        showlegend=True,
        legend=dict(title='Model/Skills', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # ...existing code...
    html.Hr(),
    html.Div([
        dcc.RadioItems(
            id="skills-radio",
            options=[
                {"label": "Skills", "value": "skills"},
                {"label": "No Skills", "value": "no_skills"}
            ],
            value="skills",
            labelStyle={"display": "inline-block", "marginRight": "20px", "fontWeight": "bold", "fontSize": "16px"}
        )
    ], style={"textAlign": "center", "marginBottom": "10px"}),
    # --- New: Three grouped bar charts for reasoning, blurring, and image type effects ---
    html.Div([
        html.H2("Model Accuracy: Reasoning Effect (Color, No Blur)"),
        dcc.Graph(id="bar-reasoning-effect"),
        html.H2("Model Accuracy: Blurring Effect (Color, Low Reasoning)"),
        dcc.Graph(id="bar-blurring-effect"),
        html.H2("Model Accuracy: Image Type Effect (No Blur, Low Reasoning)"),
        dcc.Graph(id="bar-imagetype-effect"),
    ], style={"width": "100%", "display": "block", "margin": "auto"}),
    html.Hr(),
    html.Div([
        html.H2("Effectiveness of Skills vs No Skills (Stacked Bar Chart)"),
        html.Div([
            dcc.Dropdown(
                id="stacked-category-dropdown",
                options=[
                    {"label": "Reasoning Mode", "value": "reasoning_mode"},
                    {"label": "Blur Level", "value": "blur_level"},
                    {"label": "Image Type", "value": "image_type"}
                ],
                value="reasoning_mode",
                clearable=False,
                style={"width": "250px", "display": "inline-block"}
            ),
            # ...existing code...
        ], style={"textAlign": "center", "marginBottom": "10px"}),
        dcc.Graph(id="stacked-skills-effect-bar")
    ], style={"width": "100%", "display": "block", "margin": "auto"}),
    html.Hr(),
    html.H2("Model Accuracy Heatmaps"),
    html.Div([
        html.Div([
            html.H4("gpt-4o Heatmap"),
            html.Div(id="heatmap-gpt-4o")
        ], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.H4("gpt-5.1 Heatmap"),
            html.Label("Reasoning Mode for Heatmap"),
            dcc.Dropdown(
                id="dropdown-gpt-5_1-heatmap-reasoning",
                options=reasoning_options,
                value="low",
                clearable=False
            ),
            html.Div(id="heatmap-gpt-5_1")
        ], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.H4("gpt-5.2 Heatmap"),
            html.Label("Reasoning Mode for Heatmap"),
            dcc.Dropdown(
                id="dropdown-gpt-5_2-heatmap-reasoning",
                options=reasoning_options,
                value="low",
                clearable=False
            ),
            html.Div(id="heatmap-gpt-5_2")
        ], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"width": "100%", "display": "flex", "justifyContent": "space-between"}),
    html.Br(),
    html.Hr(),
    html.H1("MS Paint Model Results Table (Appendix)", style={"marginTop": "40px"}),
    html.Div([
        html.Div([
            html.H3("gpt-4o Results"),
            html.Label("Image Type"),
            dcc.Dropdown(
                id="dropdown-image-type-gpt4o",
                options=image_type_options,
                value="color",
                clearable=False
            ),
            html.Label("Blur Level"),
            dcc.Dropdown(
                id="dropdown-blur-level-gpt4o",
                options=blur_level_options,
                value="no_blur",
                clearable=False
            ),
            html.Div(id="table-gpt-4o")
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.H3("gpt-5.1 Results"),
            html.Label("Image Type"),
            dcc.Dropdown(
                id="dropdown-image-type-gpt51",
                options=image_type_options,
                value="color",
                clearable=False
            ),
            html.Label("Blur Level"),
            dcc.Dropdown(
                id="dropdown-blur-level-gpt51",
                options=blur_level_options,
                value="no_blur",
                clearable=False
            ),
            html.Label("Reasoning Mode"),
            dcc.Dropdown(
                id="dropdown-gpt-5_1",
                options=reasoning_options,
                value="low",
                clearable=False
            ),
            html.Div(id="table-gpt-5_1")
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
        html.Div([
            html.H3("gpt-5.2 Results"),
            html.Label("Image Type"),
            dcc.Dropdown(
                id="dropdown-image-type-gpt52",
                options=image_type_options,
                value="color",
                clearable=False
            ),
            html.Label("Blur Level"),
            dcc.Dropdown(
                id="dropdown-blur-level-gpt52",
                options=blur_level_options,
                value="no_blur",
                clearable=False
            ),
            html.Label("Reasoning Mode"),
            dcc.Dropdown(
                id="dropdown-gpt-5_2",
                options=reasoning_options,
                value="low",
                clearable=False
            ),
            html.Div(id="table-gpt-5_2")
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top"}),
    ], style={"width": "100%", "display": "flex", "justifyContent": "space-between"}),
    ])
# --- Callback for stacked bar chart ---
@app.callback(
    Output('stacked-skills-effect-bar', 'figure'),
    [Input('stacked-category-dropdown', 'value')]
)
def update_stacked_skills_effect_bar(category):
    return build_stacked_skills_figure(df_results, category)

@app.callback(
    Output("bar-reasoning-effect", "figure"),
    [Input('url', 'pathname'), Input('skills-radio', 'value')]
)
def bar_reasoning_effect(_, skills_mode):
    models = ['gpt-4o', 'gpt-5.1', 'gpt-5.2']
    reasoning_modes = ['low', 'medium', 'high']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    bar_colors = {m: c for m, c in zip(models, colors)}
    bars = []
    x = []
    y = []
    bar_names = []
    for i, model in enumerate(models):
        for j, reasoning in enumerate(reasoning_modes):
            if model == 'gpt-4o' and reasoning != 'low':
                continue  # gpt-4o only has low
            df = df_results[(df_results['Model'] == model) &
                            (df_results['image_type'] == 'color') &
                            (df_results['blur_level'] == 'no_blur') &
                            (df_results['skills_mode'] == skills_mode)]
            if model != 'gpt-4o':
                df = df[df['reasoning_mode'] == reasoning]
            acc = df['Correct'].mean() if not df.empty else None
            x.append(f"{model} ({reasoning})")
            y.append(acc if acc is not None else 0)
            bar_names.append(model)
    fig = go.Figure()
    for model, color in bar_colors.items():
        model_x = [xk for xk, mk in zip(x, bar_names) if mk == model]
        model_y = [yk for yk, mk in zip(y, bar_names) if mk == model]
        fig.add_bar(x=model_x, y=model_y, name=model, marker_color=color)
    # Add vertical red dotted lines between model groups
    n_per_model = [sum(1 for mk in bar_names if mk == m) for m in models]
    sep_indices = [sum(n_per_model[:i]) - 0.5 for i in range(1, len(models))]
    for idx in sep_indices:
        fig.add_vline(x=idx, line_dash="dot", line_color="red", line_width=2)
    fig.update_layout(
        yaxis=dict(title='Accuracy', range=[0, 1]),
        xaxis=dict(title='Model (Reasoning Mode)'),
        title='Reasoning Effect: Color, No Blur',
        height=400,
        width=800,
        showlegend=True,
        legend=dict(title='Model', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

@app.callback(
    Output("bar-blurring-effect", "figure"),
    [Input('url', 'pathname'), Input('skills-radio', 'value')]
)
def bar_blurring_effect(_, skills_mode):
    models = ['gpt-4o', 'gpt-5.1', 'gpt-5.2']
    blur_levels = ['no_blur', 'med_blur', 'heavy_blur']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    bar_colors = {m: c for m, c in zip(models, colors)}
    bars = []
    x = []
    y = []
    bar_names = []
    for i, model in enumerate(models):
        for j, blur in enumerate(blur_levels):
            df = df_results[(df_results['Model'] == model) &
                            (df_results['image_type'] == 'color') &
                            (df_results['blur_level'] == blur) &
                            (df_results['skills_mode'] == skills_mode)]
            if model != 'gpt-4o':
                df = df[df['reasoning_mode'] == 'low']
            acc = df['Correct'].mean() if not df.empty else None
            x.append(f"{model} ({blur})")
            y.append(acc if acc is not None else 0)
            bar_names.append(model)
    fig = go.Figure()
    for model, color in bar_colors.items():
        model_x = [xk for xk, mk in zip(x, bar_names) if mk == model]
        model_y = [yk for yk, mk in zip(y, bar_names) if mk == model]
        fig.add_bar(x=model_x, y=model_y, name=model, marker_color=color)
    # Add vertical red dotted lines between model groups
    n_per_model = [sum(1 for mk in bar_names if mk == m) for m in models]
    sep_indices = [sum(n_per_model[:i]) - 0.5 for i in range(1, len(models))]
    for idx in sep_indices:
        fig.add_vline(x=idx, line_dash="dot", line_color="red", line_width=2)
    fig.update_layout(
        yaxis=dict(title='Accuracy', range=[0, 1]),
        xaxis=dict(title='Model (Blur Level)'),
        title='Blurring Effect: Color, Low Reasoning',
        height=400,
        width=800,
        showlegend=True,
        legend=dict(title='Model', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig

@app.callback(
    Output("bar-imagetype-effect", "figure"),
    [Input('url', 'pathname'), Input('skills-radio', 'value')]
)
def bar_imagetype_effect(_, skills_mode):
    models = ['gpt-4o', 'gpt-5.1', 'gpt-5.2']
    image_types = ['color', 'greyscale', 'inverted-greyscale']
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    bar_colors = {m: c for m, c in zip(models, colors)}
    bars = []
    x = []
    y = []
    bar_names = []
    for i, model in enumerate(models):
        for j, img_type in enumerate(image_types):
            df = df_results[(df_results['Model'] == model) &
                            (df_results['image_type'] == img_type) &
                            (df_results['blur_level'] == 'no_blur') &
                            (df_results['skills_mode'] == skills_mode)]
            if model != 'gpt-4o':
                df = df[df['reasoning_mode'] == 'low']
            acc = df['Correct'].mean() if not df.empty else None
            x.append(f"{model} ({img_type})")
            y.append(acc if acc is not None else 0)
            bar_names.append(model)
    fig = go.Figure()
    for model, color in bar_colors.items():
        model_x = [xk for xk, mk in zip(x, bar_names) if mk == model]
        model_y = [yk for yk, mk in zip(y, bar_names) if mk == model]
        fig.add_bar(x=model_x, y=model_y, name=model, marker_color=color)
    # Add vertical red dotted lines between model groups
    n_per_model = [sum(1 for mk in bar_names if mk == m) for m in models]
    sep_indices = [sum(n_per_model[:i]) - 0.5 for i in range(1, len(models))]
    for idx in sep_indices:
        fig.add_vline(x=idx, line_dash="dot", line_color="red", line_width=2)
    fig.update_layout(
        yaxis=dict(title='Accuracy', range=[0, 1]),
        xaxis=dict(title='Model (Image Type)'),
        title='Image Type Effect: No Blur, Low Reasoning',
        height=400,
        width=800,
        showlegend=True,
        legend=dict(title='Model', orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    return fig


# Callbacks for each model's dropdown to filter and display only that model's results

# Always show gpt-4o results, filtered by image type and blur level
@app.callback(
    Output("table-gpt-4o", "children"),
    [Input('dropdown-image-type-gpt4o', 'value'),
     Input('dropdown-blur-level-gpt4o', 'value')]
)
def update_table_gpt4o(image_type, blur_level):
    df = df_results[(df_results["Model"] == "gpt-4o") &
                   (df_results["image_type"] == image_type) &
                   (df_results["blur_level"] == blur_level)]
    return df_to_dash_table(df)

# Heatmap for gpt-4o (static, no dropdowns)
@app.callback(
    Output("heatmap-gpt-4o", "children"),
    [Input('url', 'pathname')]
)
def update_heatmap_gpt4o(_):
    return model_heatmap_figure(df_results, 'gpt-4o')

# Heatmap for gpt-5.1 (dropdown for reasoning)
@app.callback(
    Output("heatmap-gpt-5_1", "children"),
    [Input('dropdown-gpt-5_1-heatmap-reasoning', 'value')]
)
def update_heatmap_gpt51(reasoning_mode):
    return model_heatmap_figure(df_results, 'gpt-5.1', reasoning_mode)

# Heatmap for gpt-5.2 (dropdown for reasoning)
@app.callback(
    Output("heatmap-gpt-5_2", "children"),
    [Input('dropdown-gpt-5_2-heatmap-reasoning', 'value')]
)
def update_heatmap_gpt52(reasoning_mode):
    return model_heatmap_figure(df_results, 'gpt-5.2', reasoning_mode)

# Table for gpt-5.1
@app.callback(
    Output("table-gpt-5_1", "children"),
    [Input("dropdown-gpt-5_1", "value"),
     Input("dropdown-image-type-gpt51", "value"),
     Input("dropdown-blur-level-gpt51", "value")]
)
def update_table_gpt51(reasoning_mode, image_type, blur_level):
    df = df_results[(df_results["Model"] == "gpt-5.1") &
                   (df_results["reasoning_mode"] == reasoning_mode) &
                   (df_results["image_type"] == image_type) &
                   (df_results["blur_level"] == blur_level)]
    return df_to_dash_table(df)

# Table for gpt-5.2
@app.callback(
    Output("table-gpt-5_2", "children"),
    [Input("dropdown-gpt-5_2", "value"),
     Input("dropdown-image-type-gpt52", "value"),
     Input("dropdown-blur-level-gpt52", "value")]
)
def update_table_gpt52(reasoning_mode, image_type, blur_level):
    df = df_results[(df_results["Model"] == "gpt-5.2") &
                   (df_results["reasoning_mode"] == reasoning_mode) &
                   (df_results["image_type"] == image_type) &
                   (df_results["blur_level"] == blur_level)]
    return df_to_dash_table(df)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
