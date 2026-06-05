import plotly.express as px


def line_chart(
        df,
        x_col,
        y_col,
        title
):

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title
    )

    return fig


def bar_chart(
        df,
        x_col,
        y_col,
        title
):

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title
    )

    return fig


def scatter_chart(
        df,
        x_col,
        y_col,
        title
):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=title
    )

    return fig
