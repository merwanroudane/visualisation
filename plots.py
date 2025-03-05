import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import io

# Page configuration
st.set_page_config(page_title="Data Visualization Tool", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        color: #424242;
        font-size: 1rem;
    }
    .highlight {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>📊 Advanced Data Visualization Tool</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='info-text'>Upload your Excel file and create interactive visualizations using various libraries.</p>",
    unsafe_allow_html=True)

# Sidebar for file upload and basic settings
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        st.markdown("---")
        st.header("Visualization Settings")
        viz_library = st.selectbox(
            "Select Visualization Library",
            ["Plotly", "Matplotlib", "Seaborn"]
        )


# Main function to load and process data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df


# Function to identify data types
def identify_data_types(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = []

    # Check for potential date columns currently classified as object/string
    for col in categorical_cols:
        try:
            # Check if the column can be converted to datetime
            pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass

    return numeric_cols, categorical_cols, date_cols


# Function to get plot recommendations based on data types
def get_plot_recommendations(num_cols, cat_cols, date_cols):
    plots = []

    # Basic plots
    if len(num_cols) >= 1:
        plots.append("Histogram")
        plots.append("Box Plot")
        plots.append("Violin Plot")
        plots.append("KDE Plot")

    if len(cat_cols) >= 1:
        plots.append("Bar Chart")
        plots.append("Count Plot")
        plots.append("Pie Chart")

    # Relationship plots
    if len(num_cols) >= 2:
        plots.append("Scatter Plot")
        plots.append("Bubble Chart")
        plots.append("Heatmap (Correlation)")
        plots.append("Line Plot")
        plots.append("Area Chart")

    if len(num_cols) >= 1 and len(cat_cols) >= 1:
        plots.append("Box Plot by Category")
        plots.append("Violin Plot by Category")
        plots.append("Bar Chart (Mean/Sum by Category)")
        plots.append("Strip Plot")
        plots.append("Swarm Plot")

    # Time series plots
    if len(date_cols) >= 1:
        plots.append("Time Series Line Plot")
        plots.append("Time Series Area Plot")
        plots.append("Calendar Heatmap")
        plots.append("Candlestick Chart")

    if len(date_cols) >= 1 and len(num_cols) >= 1:
        plots.append("Time Series with Moving Average")
        plots.append("Time Series Decomposition")

    # Advanced plots
    if len(num_cols) >= 3:
        plots.append("3D Scatter Plot")
        plots.append("Parallel Coordinates")

    return sorted(plots)


# Plotting functions for different libraries
def create_plotly_visualization(df, plot_type, settings):
    fig = None

    if plot_type == "Histogram":
        fig = px.histogram(
            df, x=settings['x_column'],
            color=settings.get('color_column'),
            nbins=settings.get('bins', 20),
            opacity=settings.get('opacity', 0.7),
            marginal=settings.get('marginal', "box"),
            title=f"Histogram of {settings['x_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Box Plot":
        fig = px.box(
            df, y=settings['y_column'],
            x=settings.get('x_column'),
            color=settings.get('color_column'),
            points=settings.get('points', 'outliers'),
            title=f"Box Plot of {settings['y_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Violin Plot":
        fig = px.violin(
            df, y=settings['y_column'],
            x=settings.get('x_column'),
            color=settings.get('color_column'),
            box=settings.get('show_box', True),
            points=settings.get('points', 'outliers'),
            title=f"Violin Plot of {settings['y_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Bar Chart":
        if settings.get('agg_func'):
            agg_df = df.groupby(settings['x_column'])[settings['y_column']].agg(settings['agg_func']).reset_index()
            fig = px.bar(
                agg_df, x=settings['x_column'], y=settings['y_column'],
                color=settings.get('color_column'),
                title=f"Bar Chart of {settings['agg_func']} {settings['y_column']} by {settings['x_column']}",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
        else:
            fig = px.bar(
                df, x=settings['x_column'], y=settings['y_column'],
                color=settings.get('color_column'),
                title=f"Bar Chart of {settings['y_column']} by {settings['x_column']}",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )

    elif plot_type == "Scatter Plot":
        fig = px.scatter(
            df, x=settings['x_column'], y=settings['y_column'],
            color=settings.get('color_column'),
            size=settings.get('size_column'),
            hover_name=settings.get('hover_name'),
            trendline=settings.get('trendline'),
            title=f"Scatter Plot of {settings['y_column']} vs {settings['x_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Pie Chart":
        if settings.get('agg_func'):
            agg_df = df.groupby(settings['names_column'])[settings['values_column']].agg(
                settings['agg_func']).reset_index()
            fig = px.pie(
                agg_df, names=settings['names_column'], values=settings['values_column'],
                title=f"Pie Chart of {settings['agg_func']} {settings['values_column']} by {settings['names_column']}",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
        else:
            counts = df[settings['names_column']].value_counts().reset_index()
            counts.columns = [settings['names_column'], 'count']
            fig = px.pie(
                counts, names=settings['names_column'], values='count',
                title=f"Pie Chart of {settings['names_column']} Distribution",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )

    elif plot_type == "Line Plot":
        fig = px.line(
            df, x=settings['x_column'], y=settings['y_column'],
            color=settings.get('color_column'),
            markers=settings.get('markers', True),
            title=f"Line Plot of {settings['y_column']} vs {settings['x_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Area Chart":
        fig = px.area(
            df, x=settings['x_column'], y=settings['y_column'],
            color=settings.get('color_column'),
            title=f"Area Chart of {settings['y_column']} vs {settings['x_column']}",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Heatmap (Correlation)":
        corr_method = settings.get('corr_method', 'pearson')
        corr = df[settings['columns']].corr(method=corr_method)

        fig = px.imshow(
            corr,
            text_auto=settings.get('show_values', True),
            color_continuous_scale=settings.get('color_scale', 'RdBu_r'),
            title=f"Correlation Heatmap ({corr_method})",
            aspect="auto"
        )

    elif plot_type == "3D Scatter Plot":
        fig = px.scatter_3d(
            df, x=settings['x_column'], y=settings['y_column'], z=settings['z_column'],
            color=settings.get('color_column'),
            size=settings.get('size_column'),
            opacity=settings.get('opacity', 0.7),
            title=f"3D Scatter Plot",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    elif plot_type == "Time Series Line Plot":
        if settings.get('resample'):
            # Ensure column is datetime type
            df[settings['x_column']] = pd.to_datetime(df[settings['x_column']])
            # Resample data (e.g., daily, weekly, monthly)
            resampled_df = df.set_index(settings['x_column'])
            resampled_df = resampled_df[settings['y_column']].resample(settings['resample_rule']).agg(
                settings['agg_func'])
            resampled_df = resampled_df.reset_index()

            fig = px.line(
                resampled_df, x=settings['x_column'], y=settings['y_column'],
                title=f"Time Series of {settings['y_column']} ({settings['resample_rule']} {settings['agg_func']})",
                markers=settings.get('markers', True)
            )
        else:
            fig = px.line(
                df, x=settings['x_column'], y=settings['y_column'],
                color=settings.get('color_column'),
                title=f"Time Series of {settings['y_column']}",
                markers=settings.get('markers', True),
                color_discrete_sequence=px.colors.qualitative.Plotly
            )

    elif plot_type == "Time Series with Moving Average":
        # Ensure column is datetime type
        df[settings['x_column']] = pd.to_datetime(df[settings['x_column']])

        # Sort by date
        df_sorted = df.sort_values(by=settings['x_column'])

        # Calculate moving average
        window_size = settings.get('window_size', 7)
        df_sorted[f'MA_{window_size}'] = df_sorted[settings['y_column']].rolling(window=window_size).mean()

        fig = go.Figure()

        # Add original time series
        fig.add_trace(go.Scatter(
            x=df_sorted[settings['x_column']],
            y=df_sorted[settings['y_column']],
            mode='lines',
            name=settings['y_column'],
            line=dict(color='blue')
        ))

        # Add moving average
        fig.add_trace(go.Scatter(
            x=df_sorted[settings['x_column']],
            y=df_sorted[f'MA_{window_size}'],
            mode='lines',
            name=f'{window_size}-point Moving Average',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=f"Time Series with {window_size}-point Moving Average",
            xaxis_title=settings['x_column'],
            yaxis_title=settings['y_column'],
            legend_title="Legend"
        )

    elif plot_type == "Candlestick Chart":
        # Ensure column is datetime type and sort
        df[settings['date_column']] = pd.to_datetime(df[settings['date_column']])
        df_sorted = df.sort_values(by=settings['date_column'])

        fig = go.Figure(data=[go.Candlestick(
            x=df_sorted[settings['date_column']],
            open=df_sorted[settings['open_column']],
            high=df_sorted[settings['high_column']],
            low=df_sorted[settings['low_column']],
            close=df_sorted[settings['close_column']]
        )])

        fig.update_layout(
            title=f"Candlestick Chart",
            xaxis_title=settings['date_column'],
            yaxis_title="Price",
            xaxis_rangeslider_visible=settings.get('show_rangeslider', True)
        )

    elif plot_type == "Parallel Coordinates":
        fig = px.parallel_coordinates(
            df, color=settings['color_column'],
            dimensions=settings['dimensions'],
            title="Parallel Coordinates Plot",
            color_continuous_scale=px.colors.diverging.Tealrose
        )

    elif plot_type == "Bubble Chart":
        fig = px.scatter(
            df, x=settings['x_column'], y=settings['y_column'],
            size=settings['size_column'],
            color=settings.get('color_column'),
            hover_name=settings.get('hover_name'),
            title=f"Bubble Chart of {settings['y_column']} vs {settings['x_column']}",
            size_max=settings.get('max_bubble_size', 40),
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

    # Return the figure or None if plot type not handled
    return fig


def create_matplotlib_visualization(df, plot_type, settings):
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "Histogram":
        ax.hist(
            df[settings['x_column']],
            bins=settings.get('bins', 20),
            alpha=settings.get('opacity', 0.7),
            color=settings.get('color', 'skyblue'),
            edgecolor='black'
        )
        ax.set_xlabel(settings['x_column'])
        ax.set_ylabel('Frequency')
        ax.set_title(f"Histogram of {settings['x_column']}")

    elif plot_type == "Bar Chart":
        if settings.get('agg_func'):
            agg_df = df.groupby(settings['x_column'])[settings['y_column']].agg(settings['agg_func'])
            agg_df.plot(kind='bar', ax=ax, color=settings.get('color', 'skyblue'))
        else:
            ax.bar(
                df[settings['x_column']],
                df[settings['y_column']],
                color=settings.get('color', 'skyblue')
            )
        ax.set_xlabel(settings['x_column'])
        ax.set_ylabel(settings['y_column'])
        ax.set_title(f"Bar Chart of {settings['y_column']} by {settings['x_column']}")
        plt.xticks(rotation=45)

    elif plot_type == "Scatter Plot":
        ax.scatter(
            df[settings['x_column']],
            df[settings['y_column']],
            alpha=settings.get('opacity', 0.7),
            color=settings.get('color', 'blue')
        )
        ax.set_xlabel(settings['x_column'])
        ax.set_ylabel(settings['y_column'])
        ax.set_title(f"Scatter Plot of {settings['y_column']} vs {settings['x_column']}")

        if settings.get('trendline'):
            # Add trendline
            z = np.polyfit(df[settings['x_column']], df[settings['y_column']], 1)
            p = np.poly1d(z)
            ax.plot(df[settings['x_column']], p(df[settings['x_column']]), "r--", alpha=0.8)

    elif plot_type == "Box Plot":
        ax.boxplot(df[settings['y_column']])
        ax.set_title(f"Box Plot of {settings['y_column']}")
        ax.set_ylabel(settings['y_column'])

    elif plot_type == "Line Plot":
        ax.plot(
            df[settings['x_column']],
            df[settings['y_column']],
            marker=settings.get('marker', 'o') if settings.get('markers', True) else None,
            linestyle=settings.get('linestyle', '-'),
            color=settings.get('color', 'blue')
        )
        ax.set_xlabel(settings['x_column'])
        ax.set_ylabel(settings['y_column'])
        ax.set_title(f"Line Plot of {settings['y_column']} vs {settings['x_column']}")

    elif plot_type == "Pie Chart":
        if settings.get('agg_func'):
            agg_df = df.groupby(settings['names_column'])[settings['values_column']].agg(settings['agg_func'])
            ax.pie(
                agg_df,
                labels=agg_df.index,
                autopct='%1.1f%%',
                startangle=90
            )
        else:
            counts = df[settings['names_column']].value_counts()
            ax.pie(
                counts,
                labels=counts.index,
                autopct='%1.1f%%',
                startangle=90
            )
        ax.set_title(f"Pie Chart of {settings['names_column']}")
        ax.axis('equal')

    elif plot_type == "Heatmap (Correlation)":
        corr_method = settings.get('corr_method', 'pearson')
        corr = df[settings['columns']].corr(method=corr_method)

        im = ax.imshow(corr, cmap=settings.get('cmap', 'coolwarm'))

        # Add labels
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.columns)

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)

        # Add values in each cell
        if settings.get('show_values', True):
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                                   ha="center", va="center", color="black")

        ax.set_title(f"Correlation Heatmap ({corr_method})")

    plt.tight_layout()
    return fig


def create_seaborn_visualization(df, plot_type, settings):
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "Histogram":
        sns.histplot(
            data=df,
            x=settings['x_column'],
            bins=settings.get('bins', 20),
            kde=settings.get('kde', False),
            ax=ax,
            color=settings.get('color', 'skyblue')
        )
        ax.set_title(f"Histogram of {settings['x_column']}")

    elif plot_type == "Box Plot":
        sns.boxplot(
            data=df,
            y=settings['y_column'],
            x=settings.get('x_column'),
            hue=settings.get('hue_column'),
            ax=ax
        )
        ax.set_title(f"Box Plot of {settings['y_column']}")

    elif plot_type == "Violin Plot":
        sns.violinplot(
            data=df,
            y=settings['y_column'],
            x=settings.get('x_column'),
            hue=settings.get('hue_column'),
            split=settings.get('split', False),
            inner=settings.get('inner', 'box'),
            ax=ax
        )
        ax.set_title(f"Violin Plot of {settings['y_column']}")

    elif plot_type == "KDE Plot":
        sns.kdeplot(
            data=df,
            x=settings['x_column'],
            hue=settings.get('hue_column'),
            fill=settings.get('fill', True),
            ax=ax
        )
        ax.set_title(f"KDE Plot of {settings['x_column']}")

    elif plot_type == "Scatter Plot":
        sns.scatterplot(
            data=df,
            x=settings['x_column'],
            y=settings['y_column'],
            hue=settings.get('hue_column'),
            size=settings.get('size_column'),
            ax=ax
        )

        if settings.get('trendline'):
            sns.regplot(
                data=df,
                x=settings['x_column'],
                y=settings['y_column'],
                scatter=False,
                ax=ax,
                line_kws={"color": "red"}
            )

        ax.set_title(f"Scatter Plot of {settings['y_column']} vs {settings['x_column']}")

    elif plot_type == "Bar Chart":
        if settings.get('agg_func'):
            sns.barplot(
                data=df,
                x=settings['x_column'],
                y=settings['y_column'],
                hue=settings.get('hue_column'),
                estimator=settings['agg_func'],
                ax=ax
            )
        else:
            sns.barplot(
                data=df,
                x=settings['x_column'],
                y=settings['y_column'],
                hue=settings.get('hue_column'),
                ax=ax
            )
        ax.set_title(f"Bar Chart of {settings['y_column']} by {settings['x_column']}")
        plt.xticks(rotation=45)

    elif plot_type == "Count Plot":
        sns.countplot(
            data=df,
            x=settings['x_column'],
            hue=settings.get('hue_column'),
            ax=ax
        )
        ax.set_title(f"Count Plot of {settings['x_column']}")
        plt.xticks(rotation=45)

    elif plot_type == "Heatmap (Correlation)":
        corr_method = settings.get('corr_method', 'pearson')
        corr = df[settings['columns']].corr(method=corr_method)

        sns.heatmap(
            corr,
            annot=settings.get('show_values', True),
            cmap=settings.get('cmap', 'coolwarm'),
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(f"Correlation Heatmap ({corr_method})")

    elif plot_type == "Pair Plot":
        # For pair plots, we need to return the figure directly
        plt.close(fig)  # Close the initial figure

        fig = sns.pairplot(
            df[settings['columns']],
            hue=settings.get('hue_column'),
            diag_kind=settings.get('diag_kind', 'kde'),
            corner=settings.get('corner', False)
        )
        fig.fig.suptitle("Pair Plot", y=1.02)
        return fig.fig

    elif plot_type == "Strip Plot":
        sns.stripplot(
            data=df,
            x=settings['x_column'],
            y=settings['y_column'],
            hue=settings.get('hue_column'),
            jitter=settings.get('jitter', True),
            dodge=settings.get('dodge', False),
            ax=ax
        )
        ax.set_title(f"Strip Plot of {settings['y_column']} by {settings['x_column']}")

    elif plot_type == "Swarm Plot":
        sns.swarmplot(
            data=df,
            x=settings['x_column'],
            y=settings['y_column'],
            hue=settings.get('hue_column'),
            dodge=settings.get('dodge', False),
            ax=ax
        )
        ax.set_title(f"Swarm Plot of {settings['y_column']} by {settings['x_column']}")

    plt.tight_layout()
    return fig


# Main application logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)

    # Display basic data information
    st.markdown("<h2 class='sub-header'>📋 Data Preview</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        st.markdown(f"**Missing Values:** {df.isna().sum().sum()}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Identify data types
    numeric_cols, categorical_cols, date_cols = identify_data_types(df)

    # Get recommended plots
    recommended_plots = get_plot_recommendations(numeric_cols, categorical_cols, date_cols)

    # Plot selection
    st.markdown("<h2 class='sub-header'>📊 Create Visualization</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        selected_plot = st.selectbox("Select Plot Type", recommended_plots)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_advanced = st.checkbox("Show Advanced Options")

    # Options based on selected plot type
    plot_settings = {}

    # Common settings by plot type
    if selected_plot == "Histogram":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select column for Histogram", numeric_cols)
        with col2:
            if categorical_cols:
                color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_option != "None":
                    plot_settings['color_column'] = color_option

        if show_advanced:
            col1, col2 = st.columns(2)
            with col1:
                plot_settings['bins'] = st.slider("Number of bins", 5, 100, 20)
            with col2:
                plot_settings['opacity'] = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1)
                if viz_library == "Plotly":
                    plot_settings['marginal'] = st.selectbox("Marginal", ["box", "violin", "rug", None])

    elif selected_plot == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols)
        with col2:
            if categorical_cols:
                x_option = st.selectbox("Group by (X, optional)", ["None"] + categorical_cols)
                if x_option != "None":
                    plot_settings['x_column'] = x_option

                if len(categorical_cols) > 1:
                    color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    if color_option != "None" and color_option != x_option:
                        plot_settings['color_column'] = color_option

        if show_advanced and viz_library == "Plotly":
            plot_settings['points'] = st.selectbox("Show points", ["outliers", "suspectedoutliers", "all", False])

    elif selected_plot == "Violin Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols)
        with col2:
            if categorical_cols:
                x_option = st.selectbox("Group by (X, optional)", ["None"] + categorical_cols)
                if x_option != "None":
                    plot_settings['x_column'] = x_option

                if len(categorical_cols) > 1:
                    color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    if color_option != "None" and color_option != x_option:
                        plot_settings['color_column'] = color_option

        if show_advanced:
            if viz_library == "Plotly":
                col1, col2 = st.columns(2)
                with col1:
                    plot_settings['show_box'] = st.checkbox("Show box", True)
                with col2:
                    plot_settings['points'] = st.selectbox("Show points",
                                                           ["outliers", "suspectedoutliers", "all", False])
            elif viz_library == "Seaborn":
                col1, col2 = st.columns(2)
                with col1:
                    plot_settings['split'] = st.checkbox("Split", False)
                with col2:
                    plot_settings['inner'] = st.selectbox("Inner", ["box", "quartile", "point", "stick", None])

    elif selected_plot == "KDE Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select column for KDE", numeric_cols)
        with col2:
            if categorical_cols:
                hue_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                if hue_option != "None":
                    plot_settings['hue_column'] = hue_option

        if show_advanced and viz_library == "Seaborn":
            plot_settings['fill'] = st.checkbox("Fill", True)

    elif selected_plot == "Bar Chart":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select X column (categories)",
                                                     categorical_cols if categorical_cols else numeric_cols)
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column (values)", numeric_cols)

        col1, col2 = st.columns(2)
        with col1:
            if len(categorical_cols) > 1:
                color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_option != "None" and color_option != plot_settings['x_column']:
                    plot_settings['color_column'] = color_option
        with col2:
            agg_func = st.selectbox("Aggregation function", ["None", "mean", "sum", "count", "min", "max"])
            if agg_func != "None":
                plot_settings['agg_func'] = agg_func

    elif selected_plot == "Count Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select column for counting", categorical_cols)
        with col2:
            if len(categorical_cols) > 1:
                hue_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                if hue_option != "None" and hue_option != plot_settings['x_column']:
                    plot_settings['hue_column'] = hue_option

    elif selected_plot == "Pie Chart":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['names_column'] = st.selectbox("Select column for categories", categorical_cols)
        with col2:
            values_option = st.selectbox("Select values (optional)", ["Count"] + numeric_cols)
            if values_option != "Count":
                plot_settings['values_column'] = values_option
                agg_func = st.selectbox("Aggregation function", ["sum", "mean", "count", "min", "max"])
                plot_settings['agg_func'] = agg_func

    elif selected_plot == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select X column", numeric_cols)
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols,
                                                     index=min(1, len(numeric_cols) - 1))

        col1, col2 = st.columns(2)
        with col1:
            if categorical_cols:
                color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_option != "None":
                    plot_settings['color_column'] = color_option
        with col2:
            if len(numeric_cols) > 2:
                size_option = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                if size_option != "None":
                    plot_settings['size_column'] = size_option

        if show_advanced:
            col1, col2 = st.columns(2)
            with col1:
                plot_settings['trendline'] = st.checkbox("Show trendline", False)
            with col2:
                if viz_library == "Plotly":
                    hover_option = st.selectbox("Hover labels (optional)", ["None"] + df.columns.tolist())
                    if hover_option != "None":
                        plot_settings['hover_name'] = hover_option

    elif selected_plot == "Bubble Chart":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select X column", numeric_cols)
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols,
                                                     index=min(1, len(numeric_cols) - 1))

        col1, col2 = st.columns(2)
        with col1:
            plot_settings['size_column'] = st.selectbox("Size by", numeric_cols, index=min(2, len(numeric_cols) - 1))
        with col2:
            if categorical_cols:
                color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_option != "None":
                    plot_settings['color_column'] = color_option

        if show_advanced and viz_library == "Plotly":
            col1, col2 = st.columns(2)
            with col1:
                hover_option = st.selectbox("Hover labels (optional)", ["None"] + df.columns.tolist())
                if hover_option != "None":
                    plot_settings['hover_name'] = hover_option
            with col2:
                plot_settings['max_bubble_size'] = st.slider("Max bubble size", 10, 100, 40)

    elif selected_plot == "Line Plot":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select X column", numeric_cols + date_cols)
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols)

        if categorical_cols:
            color_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            if color_option != "None":
                plot_settings['color_column'] = color_option

        if show_advanced:
            plot_settings['markers'] = st.checkbox("Show markers", True)

    elif selected_plot == "Area Chart":
        col1, col2 = st.columns(2)
        with col1:
            plot_settings['x_column'] = st.selectbox("Select X column", numeric_cols + date_cols)
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols)

        if categorical_cols:
            color_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            if color_option != "None":
                plot_settings['color_column'] = color_option

    elif selected_plot == "Heatmap (Correlation)":
        if len(numeric_cols) < 2:
            st.error("Correlation heatmap requires at least 2 numeric columns")
        else:
            plot_settings['columns'] = st.multiselect("Select columns for correlation", numeric_cols,
                                                      default=numeric_cols[:min(6, len(numeric_cols))])

            if len(plot_settings['columns']) < 2:
                st.error("Please select at least 2 columns")

            if show_advanced:
                col1, col2 = st.columns(2)
                with col1:
                    plot_settings['corr_method'] = st.selectbox("Correlation method",
                                                                ["pearson", "kendall", "spearman"])
                with col2:
                    plot_settings['show_values'] = st.checkbox("Show correlation values", True)
                    if viz_library == "Plotly":
                        plot_settings['color_scale'] = st.selectbox("Color scale",
                                                                    ["RdBu_r", "Viridis", "Plasma", "Blues", "Reds"])
                    elif viz_library in ["Matplotlib", "Seaborn"]:
                        plot_settings['cmap'] = st.selectbox("Color map",
                                                             ["coolwarm", "viridis", "plasma", "Blues", "Reds"])

    elif selected_plot == "3D Scatter Plot":
        if len(numeric_cols) < 3:
            st.error("3D Scatter plot requires at least 3 numeric columns")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                plot_settings['x_column'] = st.selectbox("Select X column", numeric_cols)
            with col2:
                plot_settings['y_column'] = st.selectbox("Select Y column", numeric_cols,
                                                         index=min(1, len(numeric_cols) - 1))
            with col3:
                plot_settings['z_column'] = st.selectbox("Select Z column", numeric_cols,
                                                         index=min(2, len(numeric_cols) - 1))

            col1, col2 = st.columns(2)
            with col1:
                if categorical_cols:
                    color_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    if color_option != "None":
                        plot_settings['color_column'] = color_option
            with col2:
                if len(numeric_cols) > 3:
                    size_option = st.selectbox("Size by (optional)", ["None"] + numeric_cols)
                    if size_option != "None":
                        plot_settings['size_column'] = size_option

            if show_advanced:
                plot_settings['opacity'] = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1)

    elif selected_plot == "Time Series Line Plot":
        if not date_cols:
            st.error("No date columns found. Please ensure at least one column can be converted to datetime.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                plot_settings['x_column'] = st.selectbox("Select Date/Time column", date_cols)
            with col2:
                plot_settings['y_column'] = st.selectbox("Select Value column", numeric_cols)

            if categorical_cols:
                color_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                if color_option != "None":
                    plot_settings['color_column'] = color_option

            if show_advanced:
                col1, col2 = st.columns(2)
                with col1:
                    resample = st.checkbox("Resample time series", False)
                    if resample:
                        plot_settings['resample'] = True
                with col2:
                    if resample:
                        plot_settings['resample_rule'] = st.selectbox(
                            "Resampling frequency",
                            ["D", "W", "M", "Q", "Y", "H", "min"],
                            format_func=lambda x: {
                                "D": "Daily", "W": "Weekly", "M": "Monthly",
                                "Q": "Quarterly", "Y": "Yearly", "H": "Hourly",
                                "min": "Minute"
                            }[x]
                        )
                        plot_settings['agg_func'] = st.selectbox("Aggregation function",
                                                                 ["mean", "sum", "count", "min", "max"])

                plot_settings['markers'] = st.checkbox("Show markers", True)

    elif selected_plot == "Time Series with Moving Average":
        if not date_cols:
            st.error("No date columns found. Please ensure at least one column can be converted to datetime.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                plot_settings['x_column'] = st.selectbox("Select Date/Time column", date_cols)
            with col2:
                plot_settings['y_column'] = st.selectbox("Select Value column", numeric_cols)

            if show_advanced:
                plot_settings['window_size'] = st.slider("Moving Average Window Size", 2, 30, 7)

    elif selected_plot == "Candlestick Chart":
        if not date_cols:
            st.error("No date columns found. Please ensure at least one column can be converted to datetime.")
        elif len(numeric_cols) < 4:
            st.error("Candlestick chart requires at least 4 numeric columns (open, high, low, close)")
        else:
            col1, col2 = st.columns(2)
            with col1:
                plot_settings['date_column'] = st.selectbox("Select Date column", date_cols)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                plot_settings['open_column'] = st.selectbox("Select Open Price column", numeric_cols)
            with col2:
                plot_settings['high_column'] = st.selectbox("Select High Price column", numeric_cols,
                                                            index=min(1, len(numeric_cols) - 1))
            with col3:
                plot_settings['low_column'] = st.selectbox("Select Low Price column", numeric_cols,
                                                           index=min(2, len(numeric_cols) - 1))
            with col4:
                plot_settings['close_column'] = st.selectbox("Select Close Price column", numeric_cols,
                                                             index=min(3, len(numeric_cols) - 1))

            if show_advanced:
                plot_settings['show_rangeslider'] = st.checkbox("Show range slider", True)

    elif selected_plot == "Parallel Coordinates":
        if len(numeric_cols) < 3:
            st.error("Parallel coordinates plot requires at least 3 numeric columns")
        else:
            plot_settings['dimensions'] = st.multiselect("Select columns for dimensions", numeric_cols,
                                                         default=numeric_cols[:min(6, len(numeric_cols))])

            if len(plot_settings['dimensions']) < 3:
                st.error("Please select at least 3 columns")

            col_options = categorical_cols + numeric_cols
            if col_options:
                plot_settings['color_column'] = st.selectbox("Color by", col_options)

    elif selected_plot == "Pair Plot":
        if len(numeric_cols) < 2:
            st.error("Pair plot requires at least 2 numeric columns")
        else:
            plot_settings['columns'] = st.multiselect("Select columns for pair plot", numeric_cols,
                                                      default=numeric_cols[:min(4, len(numeric_cols))])

            if len(plot_settings['columns']) < 2:
                st.error("Please select at least 2 columns")

            if categorical_cols:
                hue_option = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if hue_option != "None":
                    plot_settings['hue_column'] = hue_option

            if show_advanced:
                col1, col2 = st.columns(2)
                with col1:
                    plot_settings['diag_kind'] = st.selectbox("Diagonal plot type", ["kde", "hist"])
                with col2:
                    plot_settings['corner'] = st.checkbox("Corner plot (lower triangle only)", False)

    elif selected_plot == "Strip Plot" or selected_plot == "Swarm Plot":
        col1, col2 = st.columns(2)
        with col1:
            if categorical_cols:
                plot_settings['x_column'] = st.selectbox("Select X column (categorical)", categorical_cols)
            else:
                st.error("No categorical columns found for x-axis")
        with col2:
            plot_settings['y_column'] = st.selectbox("Select Y column (numeric)", numeric_cols)

        if len(categorical_cols) > 1:
            hue_option = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            if hue_option != "None" and hue_option != plot_settings['x_column']:
                plot_settings['hue_column'] = hue_option

        if show_advanced:
            if selected_plot == "Strip Plot":
                col1, col2 = st.columns(2)
                with col1:
                    plot_settings['jitter'] = st.checkbox("Add jitter", True)
                with col2:
                    if 'hue_column' in plot_settings:
                        plot_settings['dodge'] = st.checkbox("Dodge", False)
            else:  # Swarm Plot
                if 'hue_column' in plot_settings:
                    plot_settings['dodge'] = st.checkbox("Dodge", False)

    # Create visualization button
    create_viz = st.button("Create Visualization", type="primary")

    if create_viz:
        st.markdown("<h2 class='sub-header'>🎨 Visualization Result</h2>", unsafe_allow_html=True)

        with st.spinner("Creating visualization..."):
            if viz_library == "Plotly":
                fig = create_plotly_visualization(df, selected_plot, plot_settings)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(
                        f"Could not create {selected_plot} with the current settings. Please try different options.")

            elif viz_library == "Matplotlib":
                fig = create_matplotlib_visualization(df, selected_plot, plot_settings)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error(
                        f"Could not create {selected_plot} with the current settings. Please try different options.")

            elif viz_library == "Seaborn":
                fig = create_seaborn_visualization(df, selected_plot, plot_settings)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error(
                        f"Could not create {selected_plot} with the current settings. Please try different options.")

        # Option to save the visualization
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Save Visualization")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if viz_library == "Plotly":
                save_html = st.download_button(
                    label="Download as HTML",
                    data=io.StringIO(fig.to_html()).read().encode("utf-8"),
                    file_name=f"{selected_plot.replace(' ', '_').lower()}.html",
                    mime="text/html"
                )

        with col2:
            if viz_library == "Plotly":
                # For Plotly, we need to convert to a static image
                img_bytes = fig.to_image(format="png", engine="kaleido")
                save_png = st.download_button(
                    label="Download as PNG",
                    data=img_bytes,
                    file_name=f"{selected_plot.replace(' ', '_').lower()}.png",
                    mime="image/png"
                )
            else:
                # For Matplotlib/Seaborn
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                save_png = st.download_button(
                    label="Download as PNG",
                    data=buf,
                    file_name=f"{selected_plot.replace(' ', '_').lower()}.png",
                    mime="image/png"
                )

        # Plot code display
        st.markdown("---")
        st.markdown("### Visualization Code")
        st.markdown("You can copy this code to recreate this visualization in your own environment:")

        if viz_library == "Plotly":
            code = generate_plotly_code(selected_plot, plot_settings)
            st.code(code, language="python")
        elif viz_library == "Matplotlib":
            code = generate_matplotlib_code(selected_plot, plot_settings)
            st.code(code, language="python")
        elif viz_library == "Seaborn":
            code = generate_seaborn_code(selected_plot, plot_settings)
            st.code(code, language="python")

else:
    # Show welcome screen with instructions
    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
    st.markdown("## Welcome to the Advanced Data Visualization Tool! 👋")
    st.markdown("""
    This app allows you to create interactive visualizations from your data. To get started:

    1. Upload an Excel (.xlsx) file using the file uploader in the sidebar
    2. Choose your preferred visualization library (Plotly, Matplotlib, or Seaborn)
    3. Select a visualization type from the recommended plots
    4. Configure the plot settings and create your visualization
    5. Download or export your visualization as needed

    The app will automatically detect numeric, categorical, and date columns in your data and recommend appropriate visualizations.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Sample visualizations showcase
    st.markdown("### Sample Visualizations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://plotly.com/~jackp/15042.png", caption="Interactive Scatter Plot")

    with col2:
        st.image("https://seaborn.pydata.org/_images/seaborn-heatmap-2.png", caption="Correlation Heatmap")

    with col3:
        st.image("https://plotly.com/~alexcjohnson/3482.png", caption="Time Series Analysis")


# Helper function to generate code examples
def generate_plotly_code(plot_type, settings):
    code = """import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your data
df = pd.read_excel('your_file.xlsx')

"""

    if plot_type == "Histogram":
        color_part = f", color='{settings['color_column']}'" if 'color_column' in settings else ""
        marginal_part = f", marginal='{settings['marginal']}'" if 'marginal' in settings else ""

        code += f"""# Create histogram
fig = px.histogram(
    df, x='{settings['x_column']}'{color_part},
    nbins={settings.get('bins', 20)},
    opacity={settings.get('opacity', 0.7)}{marginal_part},
    title="Histogram of {settings['x_column']}"
)

fig.show()"""

    elif plot_type == "Scatter Plot":
        color_part = f", color='{settings['color_column']}'" if 'color_column' in settings else ""
        size_part = f", size='{settings['size_column']}'" if 'size_column' in settings else ""
        hover_part = f", hover_name='{settings['hover_name']}'" if 'hover_name' in settings else ""
        trendline_part = ", trendline='ols'" if settings.get('trendline') else ""

        code += f"""# Create scatter plot
fig = px.scatter(
    df, x='{settings['x_column']}', y='{settings['y_column']}'{color_part}{size_part}{hover_part}{trendline_part},
    title="Scatter Plot of {settings['y_column']} vs {settings['x_column']}"
)

fig.show()"""

    elif plot_type == "Time Series Line Plot":
        color_part = f", color='{settings['color_column']}'" if 'color_column' in settings else ""

        if settings.get('resample'):
            code += f"""# Convert to datetime and resample
df['{settings['x_column']}'] = pd.to_datetime(df['{settings['x_column']}'])
resampled_df = df.set_index('{settings['x_column']}')
resampled_df = resampled_df['{settings['y_column']}'].resample('{settings['resample_rule']}').{settings['agg_func']}()
resampled_df = resampled_df.reset_index()

# Create time series plot
fig = px.line(
    resampled_df, x='{settings['x_column']}', y='{settings['y_column']}',
    markers={settings.get('markers', True)},
    title="Time Series of {settings['y_column']} ({settings['resample_rule']} {settings['agg_func']})"
)

fig.show()"""
        else:
            code += f"""# Convert to datetime
df['{settings['x_column']}'] = pd.to_datetime(df['{settings['x_column']}'])

# Create time series plot
fig = px.line(
    df, x='{settings['x_column']}', y='{settings['y_column']}'{color_part},
    markers={settings.get('markers', True)},
    title="Time Series of {settings['y_column']}"
)

fig.show()"""

    elif plot_type == "Time Series with Moving Average":
        code += f"""# Convert to datetime and sort
df['{settings['x_column']}'] = pd.to_datetime(df['{settings['x_column']}'])
df_sorted = df.sort_values(by='{settings['x_column']}')

# Calculate moving average
window_size = {settings.get('window_size', 7)}
df_sorted[f'MA_{{window_size}}'] = df_sorted['{settings['y_column']}'].rolling(window=window_size).mean()

# Create time series with moving average
fig = go.Figure()

# Add original time series
fig.add_trace(go.Scatter(
    x=df_sorted['{settings['x_column']}'], 
    y=df_sorted['{settings['y_column']}'],
    mode='lines',
    name='{settings['y_column']}',
    line=dict(color='blue')
))

# Add moving average
fig.add_trace(go.Scatter(
    x=df_sorted['{settings['x_column']}'], 
    y=df_sorted[f'MA_{{window_size}}'],
    mode='lines',
    name=f'{{window_size}}-point Moving Average',
    line=dict(color='red')
))

fig.update_layout(
    title=f"Time Series with {{window_size}}-point Moving Average",
    xaxis_title='{settings['x_column']}',
    yaxis_title='{settings['y_column']}',
    legend_title="Legend"
)

fig.show()"""

    elif plot_type == "Heatmap (Correlation)":
        columns_str = ", ".join([f"'{col}'" for col in settings['columns']])

        code += f"""# Calculate correlation
corr_method = '{settings.get('corr_method', 'pearson')}'
corr = df[[{columns_str}]].corr(method=corr_method)

# Create correlation heatmap
fig = px.imshow(
    corr,
    text_auto={settings.get('show_values', True)},
    color_continuous_scale='{settings.get('color_scale', 'RdBu_r')}',
    title=f"Correlation Heatmap ({{corr_method}})",
    aspect="auto"
)

fig.show()"""

    # Add more code templates for other plot types

    return code


def generate_matplotlib_code(plot_type, settings):
    code = """import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data
df = pd.read_excel('your_file.xlsx')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

"""

    if plot_type == "Histogram":
        code += f"""# Create histogram
ax.hist(
    df['{settings['x_column']}'], 
    bins={settings.get('bins', 20)},
    alpha={settings.get('opacity', 0.7)},
    color='{settings.get('color', 'skyblue')}',
    edgecolor='black'
)
ax.set_xlabel('{settings['x_column']}')
ax.set_ylabel('Frequency')
ax.set_title("Histogram of {settings['x_column']}")

plt.tight_layout()
plt.show()"""

    elif plot_type == "Scatter Plot":
        code += f"""# Create scatter plot
ax.scatter(
    df['{settings['x_column']}'], 
    df['{settings['y_column']}'],
    alpha={settings.get('opacity', 0.7)},
    color='{settings.get('color', 'blue')}'
)
ax.set_xlabel('{settings['x_column']}')
ax.set_ylabel('{settings['y_column']}')
ax.set_title("Scatter Plot of {settings['y_column']} vs {settings['x_column']}")
"""

        if settings.get('trendline'):
            code += """
# Add trendline
z = np.polyfit(df['{}'], df['{}'], 1)
p = np.poly1d(z)
ax.plot(df['{}'], p(df['{}']), "r--", alpha=0.8)
""".format(settings['x_column'], settings['y_column'], settings['x_column'], settings['x_column'])

        code += """
plt.tight_layout()
plt.show()"""

    # Add more code templates for other plot types

    return code


def generate_seaborn_code(plot_type, settings):
    code = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_theme(style="whitegrid")

# Load your data
df = pd.read_excel('your_file.xlsx')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

"""

    if plot_type == "Histogram":
        kde_part = ", kde=True" if settings.get('kde', False) else ""

        code += f"""# Create histogram
sns.histplot(
    data=df, 
    x='{settings['x_column']}',
    bins={settings.get('bins', 20)}{kde_part},
    color='{settings.get('color', 'skyblue')}',
    ax=ax
)
ax.set_title("Histogram of {settings['x_column']}")

plt.tight_layout()
plt.show()"""

    elif plot_type == "Box Plot":
        hue_part = f", hue='{settings['hue_column']}'" if 'hue_column' in settings else ""
        x_part = f", x='{settings['x_column']}'" if 'x_column' in settings else ""

        code += f"""# Create box plot
sns.boxplot(
    data=df, 
    y='{settings['y_column']}'{x_part}{hue_part},
    ax=ax
)
ax.set_title("Box Plot of {settings['y_column']}")

plt.tight_layout()
plt.show()"""

    elif plot_type == "Violin Plot":
        hue_part = f", hue='{settings['hue_column']}'" if 'hue_column' in settings else ""
        x_part = f", x='{settings['x_column']}'" if 'x_column' in settings else ""
        split_part = ", split=True" if settings.get('split', False) else ""
        inner_part = f", inner='{settings.get('inner', 'box')}'" if 'inner' in settings else ""

        code += f"""# Create violin plot
sns.violinplot(
    data=df, 
    y='{settings['y_column']}'{x_part}{hue_part}{split_part}{inner_part},
    ax=ax
)
ax.set_title("Violin Plot of {settings['y_column']}")

plt.tight_layout()
plt.show()"""

    elif plot_type == "Heatmap (Correlation)":
        columns_str = ", ".join([f"'{col}'" for col in settings['columns']])

        code += f"""# Calculate correlation
corr_method = '{settings.get('corr_method', 'pearson')}'
corr = df[[{columns_str}]].corr(method=corr_method)

# Create correlation heatmap
sns.heatmap(
    corr,
    annot={settings.get('show_values', True)},
    cmap='{settings.get('cmap', 'coolwarm')}',
    fmt='.2f',
    linewidths=0.5,
    ax=ax
)
ax.set_title(f"Correlation Heatmap ({{corr_method}})")

plt.tight_layout()
plt.show()"""

    elif plot_type == "Pair Plot":
        hue_part = f", hue='{settings['hue_column']}'" if 'hue_column' in settings else ""
        columns_str = ", ".join([f"'{col}'" for col in settings['columns']])
        diag_part = f", diag_kind='{settings.get('diag_kind', 'kde')}'" if 'diag_kind' in settings else ""
        corner_part = ", corner=True" if settings.get('corner', False) else ""

        code = """import pandas as pd
import seaborn as sns

# Set the style
sns.set_theme(style="whitegrid")

# Load your data
df = pd.read_excel('your_file.xlsx')

"""

        code += f"""# Create pair plot
pair_plot = sns.pairplot(
    df[[{columns_str}]]{hue_part}{diag_part}{corner_part}
)
pair_plot.fig.suptitle("Pair Plot", y=1.02)

plt.tight_layout()
plt.show()"""

    # Add more code templates for other plot types

    return code
st.markdown("---")
st.caption("Dr Merwan Roudane")
