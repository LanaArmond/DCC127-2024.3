import plotly.graph_objs as go
import plotly.offline as pyo
import scipy
from scipy.stats import norm
from wordcloud import WordCloud
from scipy.cluster import hierarchy
import scipy.stats as stats
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import kurtosis, skew
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# from mlxtend.frequent_patterns import apriori, association_rules

def boxplot_with_quartiles(data_row, yscale='linear'):
    # Calculate quartiles
    q1 = np.percentile(data_row, 25)
    q2 = np.percentile(data_row, 50)
    q3 = np.percentile(data_row, 75)

    # Create a boxplot
    plt.boxplot([data_row[data_row <= q1], data_row[(q1 < data_row) & (data_row <= q2)],
                 data_row[(q2 < data_row) & (data_row <= q3)], data_row[q3 < data_row]],
                labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Add labels and a title
    plt.xlabel('Quartiles')
    plt.ylabel('Values')
    plt.yscale(yscale)
    plt.title('Boxplot with Quartiles')

    # Show the plot
    plt.show()

def create_3d_scatter_plot(x_row, y_row, z_row, color='blue'):
    # Create a scatter plot trace
    trace = go.Scatter3d(
        x=x_row,
        y=y_row,
        z=z_row,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color='blue'
        )
    )

    # Create a layout
    layout = go.Layout(
        scene=dict(
            xaxis_title=x_row.name,
            yaxis_title=y_row.name,
            zaxis_title=z_row.name
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Create a figure
    fig = go.Figure(data=[trace], layout=layout)

    # Show the interactive plot in a web browser
    pyo.plot(fig, filename='interactive_3d_scatter.html')

def change_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column])

def print_simple_metrics(row):
    print(f"\n--------------------------------------\n")
    print(f"métricas gerais para {row.name}")
    print(f"média: {row.mean()}")
    print(f"mediana: {row.median()}")
    print(f"desvio padrão: {row.std()}")
    print(f"kurtosis: {kurtosis(row.dropna())}")  # Drop NA for kurtosis calculation
    print(f"skewness: {skew(row.dropna())}")  # Drop NA for skewness calculation
    print(f"\n--------------------------------------\n")
def create_filtered_histograms(row, log, filters=None, color='blue', bins=100):
    if filters is None:
        filters = [
            np.percentile(row, 100),
            np.percentile(row, 75),
            np.percentile(row, 50),
            np.percentile(row, 25),
        ]

    print(f"\n--------------------------------------\n")
    print(f"Histogramas para a coluna {row.name} com log: {log} e bins: {bins}")
    for filter in filters:
        if filter is not None:
            print(f"{row.name} <= {filter}")
            plt.hist(row[row <= filter], bins=bins, color=color, log=log)
            plt.show()
        else:
            print(f"sem filtros")
            plt.hist(row, bins=bins, color=color, log=log)
            plt.show()
        print(f"\n--------------------------------------\n")

def create_filtered_scatters(x, y, log, filters=None, color='blue'):
    if filters is None:
        filters = [
            np.percentile(x, 100),
            np.percentile(x, 75),
            np.percentile(x, 50),
            np.percentile(x, 25),
        ]

    scale = 'log' if log else 'linear'
    print(f"\n--------------------------------------\n")
    print(f"Scatterplots para colunas x: {x.name}, y: {y.name} com y em scala {scale}")
    for filter in filters:
        if filter is not None:
            print(f"removendo acima de {filter}")
            plt.figure(figsize=(8, 6))
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.yscale(scale)
            plt.scatter(x[x <= filter], y[x <= filter], c=color, marker='o', label='ads')
            plt.show()
        else:
            print(f"sem filtros")
            plt.figure(figsize=(8, 6))
            plt.xlabel(x.name)
            plt.ylabel(y.name)
            plt.yscale(scale)
            plt.scatter(x[x <= filter], y[x <= filter], c=color, marker='o', label='ads')
            plt.show()
        print(f"\n--------------------------------------\n")

def pearson_correlation(x, y):
    print(f"correlações de Pearson para {x.name} e {y.name} para os quartis de x e y")

    quartils_x = {
        '1º quartil': (np.percentile(x, 0), np.percentile(x, 25)),
        '2º quartil': (np.percentile(x, 25), np.percentile(x, 50)),
        '3º quartil': (np.percentile(x, 50), np.percentile(x, 75)),
        '4º quartil': (np.percentile(x, 75), np.percentile(x, 100)),
        'tudo': (np.percentile(x, 0), np.percentile(x, 100)),
    }
    pearson_df = []
    for quartil, q_x in quartils_x.items():
        x_filtered = x[(x >= q_x[0]) & (x <= q_x[1])]
        y_filtered = y[(x >= q_x[0]) & (x <= q_x[1])]

        if len(x[(x > q_x[0]) & (x <= q_x[1])]) > 0:
            pearson, p_value = scipy.stats.pearsonr(x_filtered, y_filtered)
            pearson_df.append({
                'pearson': pearson,
                'p_value': p_value,
                'quartil': quartil,
                'filtros': f"X -> ({q_x[0]} < x <= {q_x[1]})"
            })

    return pd.DataFrame(pearson_df)

def check_distribution_hypothesis(dist, hypothesis_dist, hypothesis_percentage):
    total = len(dist)
    subtotal = len(hypothesis_dist)

    SE = ((hypothesis_percentage * (1 - hypothesis_percentage)) / total) ** (1 / 2)
    Z = ((subtotal / total) - hypothesis_percentage) / SE

    p_value = norm(hypothesis_percentage, SE).cdf(Z)

    return p_value, Z

def get_rekogtion_dfs(ads_dict_df):
    all_img_labels_df = []
    img_labels_df = []
    for index, row in ads_dict_df.iterrows():
        img_labels_df.append({'labels': [], 'confidence': []})
        for img_data in row['rekognition']['data']:
            img_labels_df[index]['labels'].append(img_data['Name'])
            img_labels_df[index]['confidence'].append(img_data['Confidence'])
            all_img_labels_df.append({'name': img_data['Name'], 'confidence': img_data['Confidence']})

    return pd.DataFrame(img_labels_df), pd.DataFrame(all_img_labels_df)

def rekogtion_label_wordcloud(label_counts, N=100):
    top_labels = label_counts.head(N)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_labels)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='lanczos')
    plt.title("Rekognition labels WordCloud")
    plt.axis('off')
    plt.show()

def get_label_matrix(img_labels_df, label_counts, N):
    top_labels = label_counts.head(N).index.tolist()
    return pd.get_dummies(img_labels_df['labels'].explode())[top_labels].groupby(by='index', level=0).sum()

def get_heatmap(correlation_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, center=0, cmap="coolwarm")
    plt.xlabel('Labels')
    plt.ylabel('labels')
    plt.title('heatmap para correlação de labels')
    plt.show()

def get_clustermap(correlation_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    clustermap = sns.clustermap(correlation_matrix, center=0, cmap="coolwarm")
    plt.xlabel('Labels')
    plt.ylabel('labels')
    plt.title('clustermap para correlação de labels')
    plt.show()

def get_hierarchy_plot(correlation_matrix):
    plt.figure(figsize=(20, 10))
    linkage = hierarchy.linkage(correlation_matrix, method='average')
    dendrogram = hierarchy.dendrogram(linkage, labels=correlation_matrix.columns)
    plt.title("Dendrogram of Label Correlation")
    plt.ylabel("Distance")
    plt.xlabel("Labels")
    plt.xticks(fontsize=10)

    for idx, color in enumerate(dendrogram['leaves_color_list']):
        plt.gca().xaxis.get_major_ticks()[idx].label.set_color(color)

    plt.show()

def get_pca_dimensionality_reduction_2d(correlation_matrix, label_counts, N=100):
    label_counts = label_counts.head(N)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(correlation_matrix)

    plt.figure(figsize=(40, 24))

    labels = correlation_matrix.columns

    num_clusters = 15
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(correlation_matrix)
    cluster_colors = [
        'red', 'green', 'blue', 'orange', 'purple',
        'pink', 'brown', 'cyan', 'magenta', 'lime',
        'indigo', 'yellow', 'gray', 'black', 'darkred'
    ]

    min_dot_size = 300
    max_dot_size = 800
    scaled_sizes = (
            (label_counts - label_counts.min())
            / (label_counts.max() - label_counts.min())
            * (max_dot_size - min_dot_size)
            + min_dot_size
    )

    for i, label in enumerate(labels):
        plt.scatter(
            reduced_data[i, 0],
            reduced_data[i, 1],
            s=scaled_sizes[i],
            c=cluster_colors[cluster_labels[i]],
            label=f"Cluster {cluster_labels[i]}",
        )
        plt.annotate(
            label,
            (reduced_data[i, 0], reduced_data[i, 1]),
            fontsize=14,
            # c=cluster_colors[cluster_labels[i]],
            ha='left',
            va='center',
        )

    plt.xlabel("Principal Componente 1")
    plt.ylabel("Principal Componente 2")
    plt.title("Redução de dimensionalidade usando PCA 2D")
    plt.show()

def get_pca_dimensionality_reduction_3d(correlation_matrix, label_counts, N=100):
    label_counts = label_counts.head(N)
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=3)  # Choose 3 components for 3D
    reduced_data = pca.fit_transform(correlation_matrix)

    num_clusters = 15
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(correlation_matrix)
    cluster_colors = [
        'red', 'green', 'blue', 'orange', 'purple',
        'pink', 'brown', 'cyan', 'magenta', 'lime',
        'indigo', 'yellow', 'gray', 'black', 'darkred'
    ]

    labels = correlation_matrix.columns

    min_dot_size = 10
    max_dot_size = 50
    scaled_sizes = (
            (label_counts - label_counts.min())
            / (label_counts.max() - label_counts.min())
            * (max_dot_size - min_dot_size)
            + min_dot_size
    )

    scatter_df = pd.DataFrame({
        'PC1': reduced_data[:, 0],
        'PC2': reduced_data[:, 1],
        'PC3': reduced_data[:, 2],
        'Cluster': [f'Cluster {i}' for i in cluster_labels],
        'Label': labels,
        'Frequency': label_counts,
        'DotSize': scaled_sizes
    })

    fig = go.Figure()

    for i, cluster in enumerate(scatter_df['Cluster'].unique()):
        cluster_data = scatter_df[scatter_df['Cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            z=cluster_data['PC3'],
            mode='markers',
            marker=dict(
                size=cluster_data['DotSize'],
                opacity=0.7,
                color=cluster_colors[i],
                colorbar=dict(title='Cluster'),
                colorscale='Viridis',
            ),
            text=cluster_data[['Label', 'Frequency']].astype(str).agg(', '.join, axis=1),
            name=cluster,
        ))

    fig.update_layout(
        title="Interactive 3D Scatter Plot with 15 Clusters",
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3',
        ),
    )

    pyo.plot(fig, filename='interactive_3d_scatter.html')

def spearman_correlation(x, y, plot=True):
    quartils_x = {
        '1st quartile': (np.percentile(x, 0), np.percentile(x, 25)),
        '2nd quartile': (np.percentile(x, 25), np.percentile(x, 50)),
        '3rd quartile': (np.percentile(x, 50), np.percentile(x, 75)),
        '4th quartile': (np.percentile(x, 75), np.percentile(x, 100)),
        'Overall': (np.percentile(x, 0), np.percentile(x, 100)),
    }

    spearman_df = []
    for quartil, q_x in quartils_x.items():
        x_filtered = x[(x > q_x[0]) & (x <= q_x[1])]
        y_filtered = y[(x > q_x[0]) & (x <= q_x[1])]

        if len(x_filtered) > 1:
            spearman, p_value = scipy.stats.spearmanr(x_filtered, y_filtered)
            spearman_df.append({
                'Spearman': spearman,
                'P-Value': p_value,
                'Quartile': quartil,
                'Filter': f"X: {q_x[0]:.2f} <= x <= {q_x[1]:.2f}"
            })

    spearman_df = pd.DataFrame(spearman_df)

    if plot:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(spearman_df['Quartile'], spearman_df['Spearman'], alpha=0.7)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

        for bar, p_value in zip(bars, spearman_df['P-Value']):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{p_value:.3f}",
                ha='center', va='bottom', fontsize=10, color='black'
            )

        plt.title(f"Spearman Correlation by Quartile for {x.name} and {y.name}", fontsize=14)
        plt.xlabel("Quartile", fontsize=12)
        plt.ylabel("Spearman Correlation", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return spearman_df

def one_hot_encode_categories(df, column_name, add_prefix=False):
    # If the column contains only 0s and 1s (binary encoded), return the df without changes
    if df[column_name].dropna().isin([0, 1]).all():
        result = df
        unique_categories = [column_name]
    else:
        # Initialize a set to collect all unique categories
        all_categories = set()

        # Iterate through each value in the column and add categories to the set
        for value in df[column_name].dropna():
            if isinstance(value, list):  # If the value is a list, add each item to the set
                all_categories.update(value)
            else:  # Otherwise, treat it as a single category
                all_categories.add(value)

        all_categories = sorted(all_categories)  # Sort categories alphabetically (or by any other criteria)

        # Create a one-hot encoded DataFrame with 0s
        one_hot = pd.DataFrame(0, index=df.index, columns=all_categories)

        # Fill in the one-hot encoded DataFrame
        for idx, row in df.iterrows():
            value = row[column_name]
            if isinstance(value, list):  # If it's a list, mark all categories in the list as 1
                for category in value:
                    if category in one_hot.columns:
                        one_hot.at[idx, category] = 1
            else:  # If it's a single value, mark that category as 1
                if value in one_hot.columns:
                    one_hot.at[idx, value] = 1

        # Concatenate the original df with the one-hot encoded DataFrame
        result = pd.concat([df, one_hot], axis=1)

        # If add_prefix is True, add the prefix to the column names
        if add_prefix:
            result.columns = [f"{column_name}_{col}" if col in one_hot.columns else col for col in result.columns]
            unique_categories = [f"{column_name}_{cat}" for cat in one_hot.columns]
        else:
            unique_categories = one_hot.columns.tolist()

    return result, unique_categories

def category_correlation(df, category_column, performance_column):
    result_df, categories = one_hot_encode_categories(df, category_column)
    for category in categories:
        spearman_correlation(result_df[performance_column], result_df[category])

def category_correlation_all(df, categorical_columns, performance_column, top_n=5):
        correlation_summary = []

        for column in categorical_columns:
            result_df, categories = one_hot_encode_categories(df, column)

            category_counts = result_df[categories].sum(axis=0).sort_values(ascending=False).head(top_n)
            top_categories = category_counts.index

            for category in top_categories:
                spearman_df = spearman_correlation(result_df[performance_column], result_df[category], plot=False)

                for _, row in spearman_df.iterrows():
                    correlation_summary.append({
                        'Category': category,
                        'Quartile': row['Quartile'],
                        'Correlation': row['Spearman'],
                        'P-Value': row['P-Value'],
                        'Original Column': column
                    })

        correlation_df = pd.DataFrame(correlation_summary)
        correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

        correlation_df['Hover Text'] = (
                "Original Column: " + correlation_df['Original Column'] + "<br>" +
                "Quartile: " + correlation_df['Quartile'].str[:1] + "<br>" +
                "Correlation: " + correlation_df['Correlation'].round(2).astype(str) + "<br>" +
                "P-value: " + correlation_df['P-Value'].round(3).astype(str)
        )
        correlation_df['Label'] = (correlation_df['Category'] + ' - ' + correlation_df['Quartile'].str[:1] + ' - ' + correlation_df['Original Column'].str[:3])

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=correlation_df['Label'],
            y=correlation_df['Correlation'],
            text=correlation_df['Hover Text'],
            hoverinfo='text',
            marker=dict(color=correlation_df['Correlation'], colorscale='Viridis', showscale=True),
            name='Spearman Correlation'
        ))

        fig.update_layout(
            xaxis=dict(range=[0, min(10, len(correlation_df))]),
            showlegend=True,
        )

        fig.show()

        return correlation_df

def encode_top_categories(df, categorical_columns, top_n):
    encoded_data = pd.DataFrame()
    all_categories = []

    for col in categorical_columns:
        result_df, categories = one_hot_encode_categories(df, col, add_prefix=True)

        category_counts = result_df[categories].sum(axis=0).sort_values(ascending=False).head(top_n)
        top_categories = category_counts.index.tolist()

        all_categories.extend(top_categories)

        encoded_data = pd.concat([encoded_data, result_df[top_categories]], axis=1)

    return encoded_data, all_categories

def categories_pairing_corr(df, categorical_columns, top_n=5):
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)

    correlation_matrix = pd.DataFrame(index=all_categories, columns=all_categories, dtype=float)

    for cat1 in all_categories:
        for cat2 in all_categories:

            if cat1 == cat2:
                correlation_matrix.loc[cat1, cat2] = 1.0
            else:
                corr, _ = stats.spearmanr(encoded_data[cat1], encoded_data[cat2])
                correlation_matrix.loc[cat1, cat2] = corr


    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1, cbar=True)
    plt.title("Cramér's V Correlation Heatmap of Categories")
    plt.show()

    return correlation_matrix

def plot_most_common_correlations_graph(correlation_matrix, top_n=20):
    common_corr = []

    seen_pairs = set()  # To store unique pairs

    for cat1 in correlation_matrix.columns:
        for cat2 in correlation_matrix.columns:
            if cat1 != cat2:
                # Create a sorted pair (ensures (A, B) and (B, A) are considered the same)
                pair = tuple(sorted([cat1, cat2]))

                # Only process the pair if it hasn't been seen before
                if pair not in seen_pairs:
                    cramer_value = correlation_matrix.loc[cat1, cat2]

                    # If correlation is non-zero, classify as common
                    if cramer_value > 0:
                        common_corr.append((cat1, cat2, cramer_value))
                        seen_pairs.add(pair)  # Mark this pair as seen

    # Step 3: Sort the correlations by their values (descending)
    common_corr.sort(key=lambda x: x[2], reverse=True)

    # Step 4: Select the top N common pairs
    top_common = common_corr[:top_n]

    # Step 5: Build the Graph
    G = nx.Graph()

    # Add edges to the graph
    for cat1, cat2, cramer_value in top_common:
        G.add_edge(cat1, cat2, weight=cramer_value)

    # Step 6: Calculate edge colors based on the correlation intensity
    edge_colors = [data['weight'] for u, v, data in G.edges(data=True)]

    # Normalize the edge colors to map them to a color scale
    min_corr = min(edge_colors)
    max_corr = max(edge_colors)
    norm = plt.Normalize(min_corr, max_corr)
    cmap = plt.get_cmap("coolwarm")  # You can choose any colormap like 'viridis', 'plasma', etc.
    edge_color_values = [cmap(norm(val)) for val in edge_colors]

    # Step 7: Visualize the graph using a spring layout, adjusting layout based on correlation intensity
    # Use the weight as a factor to influence the node positions, scale by the correlation intensity
    k_values = [1 / (cramer_value + 1e-5) for _, _, cramer_value in top_common]  # Inverse of correlation intensity to scale distance
    k_avg = sum(k_values) / len(k_values)  # Average k-value to adjust the overall layout
    pos = nx.spring_layout(G, seed=42, k=k_avg, weight='weight')  # Positions for nodes (with dynamic distance scaling)

    plt.figure(figsize=(12, 10))

    # Draw the nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightgray', alpha=0.8)  # Nodes color as light gray
    nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_color_values, alpha=0.7)

    # Adjust font size for labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')  # Reduced font size

    # Title and display
    plt.title("Most Common Categorical Correlations (Network Graph)", size=16)
    plt.axis('off')  # Hide the axis for better aesthetics

    # Create the colorbar and associate it with the edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array for the ScalarMappable

    # Create a new axis for the colorbar and set its position
    cbar_ax = plt.gca().inset_axes([1.05, 0.15, 0.03, 0.7])  # Position for colorbar (adjust as necessary)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Correlation Intensity")

    # Step 8: Add a color intensity legend for correlation values
    legend_labels = [f'{round(min_corr + (i * (max_corr - min_corr) / 3), 2)}' for i in range(4)]
    legend_colors = [cmap(norm(min_corr + (i * (max_corr - min_corr) / 3))) for i in range(4)]
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=10, label=label)
                       for col, label in zip(legend_colors, legend_labels)]

    plt.legend(handles=legend_elements, title="Correlation Intensity", loc='upper left', bbox_to_anchor=(1.05, 1))

    # Show the graph
    plt.show()

def pca_and_cluster(df, categorical_columns, numerical_columns, top_n=3, n_clusters=3, random_state=42, scale=True, logy=False, logx=False):
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)
    encoded_data = pd.concat([encoded_data, df[numerical_columns]], axis=1)

    if scale:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(encoded_data)
    else:
        scaled_data = encoded_data

    pca = PCA(n_components=min(5, len(all_categories)))
    pca_data = pca.fit_transform(scaled_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(pca_data)

    pca_df = df.copy()
    pca_df['pca1'] = pca_data[:, 0]
    pca_df['pca2'] = pca_data[:, 1]
    pca_df['cluster'] = clusters

    loadings = pd.DataFrame(
        pca.components_,
        columns=encoded_data.columns,
        index=[f'PCA{i + 1}' for i in range(pca.n_components_)]
    )

    variable_correlations = {
        f"PCA{i + 1}": loadings.iloc[i]
        .abs()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
        for i in range(pca.n_components_)
    }

    explained_variance_ratio = pca.explained_variance_ratio_
    pca_contributions = (explained_variance_ratio[:5]).round(4)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['pca1'], pca_df['pca2'], c=pca_df['cluster'], cmap='viridis', s=50)
    plt.title("PCA Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")

    if logy:
        plt.yscale('log')

    if logx:
        plt.xscale('log')

    plt.show()

    return pca_df, variable_correlations, pca_contributions

def cluster_numeric_summary(df, cluster_col, numeric_cols):
    summary = []
    for cluster, group in df.groupby(cluster_col):
        cluster_metrics = {'cluster': cluster}
        cluster_metrics['total_entries'] = len(group)

        for col in numeric_cols:
            data = group[col].dropna()
            cluster_metrics[f'{col}_mean'] = data.mean()
            cluster_metrics[f'{col}_median'] = data.median()
            cluster_metrics[f'{col}_variance'] = data.var()
            cluster_metrics[f'{col}_std_dev'] = data.std()
            cluster_metrics[f'{col}_min'] = data.min()
            cluster_metrics[f'{col}_max'] = data.max()
            cluster_metrics[f'{col}_iqr'] = data.quantile(0.75) - data.quantile(0.25)
            cluster_metrics[f'{col}_count'] = data.count()
            cluster_metrics[f'{col}_skewness'] = skew(data)
            cluster_metrics[f'{col}_kurtosis'] = kurtosis(data)

        summary.append(cluster_metrics)

    return pd.DataFrame(summary)

def cluster_categorical_summary(df, cluster_col, categorical_columns, top_n=3):
    """
    Compute the percentage distribution of the top N categories for each cluster.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        cluster_col (str): The column name representing cluster labels.
        categorical_columns (list): List of categorical column names.
        top_n (int): Number of top categories to consider for each column.

    Returns:
        pd.DataFrame: Summary of top categories' percentages for each cluster.
    """
    # Use the provided function to encode top categories
    encoded_df, top_categories = encode_top_categories(df, categorical_columns, top_n)

    # Add the cluster column back to the encoded DataFrame
    encoded_df[cluster_col] = df[cluster_col]

    # Compute the percentage distribution for each cluster
    summary = (
            encoded_df.groupby(cluster_col)
            .sum()
            .div(encoded_df.groupby(cluster_col).size(), axis=0)
            * 100
    )

    return summary

def find_discriminative_metrics(df, numeric_columns, categorical_columns, cluster_col, top_n=3):
    encoded_df, top_categories = encode_top_categories(df, categorical_columns, top_n)

    combined_df = pd.concat([df[numeric_columns + [cluster_col]], encoded_df], axis=1)

    results = []

    for col in numeric_columns:
        clusters = [combined_df[combined_df[cluster_col] == c][col].dropna() for c in combined_df[cluster_col].unique()]
        stat, p_value = stats.kruskal(*clusters)
        test_type = "Kruskal-Wallis"

        results.append({
            "column": col,
            "statistic": stat,
            "p_value": p_value,
            "test": test_type
        })

    for col in encoded_df.columns:
        contingency_table = pd.crosstab(combined_df[cluster_col], combined_df[col])
        stat, p_value, _, _ = stats.chi2_contingency(contingency_table)

        results.append({
            "column": col,
            "statistic": stat,
            "p_value": p_value,
            "test": "Chi-Square"
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    top_3 = results_df.head(3)
    return top_3

def random_forest_classification( df, categorical_columns, numerical_column, top_n=3, test_size=0.2, random_state=42, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1):
    # Encode categorical data
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)
    encoded_data = pd.concat([encoded_data, df[numerical_column]], axis=1)

    def categorize_numerical(values):
        transformed_values = np.log1p(values)
        bins = np.percentile(transformed_values, [0, 50, 100])
        labels = ['low', 'high']
        return pd.cut(transformed_values, bins=bins, labels=labels, include_lowest=True)


    # Categorize performance column
    df['performance_category'] = categorize_numerical(df[numerical_column])

    # Encode the target variable into numerical labels for dtreeviz
    label_encoder = LabelEncoder()
    df['performance_category_encoded'] = label_encoder.fit_transform(df['performance_category'])

    X = encoded_data[all_categories]
    y = df['performance_category_encoded']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Initialize and train the RandomForest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    cm = metrics["confusion_matrix"]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar=False, square=True, annot_kws={"size": 16})

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Feature Importance Analysis
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": all_categories,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)


    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", hue="Feature")
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return model, metrics, cm, df, feature_importance_df

def apriori_associations(df, categorical_columns, itemsets=20, top_n=5, min_support=0.1, metric="lift", min_threshold=1.0):
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)
    encoded_data = encoded_data[all_categories]
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold, num_itemsets=itemsets)
    sorted_rules = rules.sort_values(by=metric, ascending=False)

    return sorted_rules, frequent_itemsets

def random_forest_classification(
        df,
        categorical_columns,
        numerical_column,
        top_n=3,
        test_size=0.2,
        random_state=42,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
):
    # Encode categorical data
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)
    encoded_data = pd.concat([encoded_data, df[numerical_column]], axis=1)

    def categorize_numerical(values):
        transformed_values = np.log1p(values)
        bins = np.percentile(transformed_values, [0, 50, 100])
        labels = ['low', 'high']
        return pd.cut(transformed_values, bins=bins, labels=labels, include_lowest=True)


    # Categorize performance column
    df['performance_category'] = categorize_numerical(df[numerical_column])

    # Encode the target variable into numerical labels for dtreeviz
    label_encoder = LabelEncoder()
    df['performance_category_encoded'] = label_encoder.fit_transform(df['performance_category'])

    X = encoded_data[all_categories]
    y = df['performance_category_encoded']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Initialize and train the RandomForest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    cm = metrics["confusion_matrix"]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                cbar=False, square=True, annot_kws={"size": 16})

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Feature Importance Analysis
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": all_categories,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)


    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", hue="Feature")
    plt.title("Feature Importance in Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    return model, metrics, cm, df, feature_importance_df

def plot_association_network(sorted_rules, top_n=10):
    # Filtra as top_n regras
    top_rules = sorted_rules.head(top_n)

    # Cria o grafo
    G = nx.DiGraph()

    for _, row in top_rules.iterrows():
        antecedents = tuple(row['antecedents'])
        consequents = tuple(row['consequents'])
        G.add_edge(antecedents, consequents, weight=row['lift'])

    # Desenha o grafo
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8)

    # Adiciona os pesos das arestas (lift)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Association Rules Network", fontsize=14)
    plt.show()

def apriori_numerical_associations(
        df,
        categorical_columns,
        numerical_column,
        itemsets=20,
        top_n=5,
        min_support=0.1,
        metric="lift",
        min_threshold=1.0
):
    binned_numerical = pd.qcut(df[numerical_column], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates='drop')
    df[f"{numerical_column}_binned"] = binned_numerical
    encoded_data, all_categories = encode_top_categories(df, categorical_columns, top_n)
    encoded_data = pd.concat([encoded_data, pd.get_dummies(df[f"{numerical_column}_binned"])], axis=1)
    frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold, num_itemsets=itemsets)
    categories = ["Q1", "Q2", "Q3", "Q4"]
    rules_with_numerical = rules[
        rules['antecedents'].apply(lambda x: any(item in categories for item in x)) |
        rules['consequents'].apply(lambda x: any(item in categories for item in x))
        ]
    sorted_rules = rules_with_numerical.sort_values(by=metric, ascending=False).head(itemsets)
    return sorted_rules, frequent_itemsets