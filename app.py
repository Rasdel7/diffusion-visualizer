import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diffusion Model Visualizer",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Diffusion Model Visualizer")
st.markdown("Understand DDPM, DDIM and Stable "
            "Diffusion — the models behind "
            "DALL-E, Midjourney and Stable Diffusion.")
st.markdown("---")

# ── Diffusion math ────────────────────────────
def get_beta_schedule(T=1000,
                       schedule='linear'):
    if schedule == 'linear':
        return np.linspace(1e-4, 0.02, T)
    elif schedule == 'cosine':
        s = 0.008
        steps = np.linspace(0, T, T+1)
        alpha_bar = np.cos(
            (steps/T + s) / (1+s) *
            np.pi/2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / \
                    alpha_bar[:-1]
        return np.clip(betas, 0, 0.999)
    elif schedule == 'quadratic':
        return np.linspace(
            1e-4**0.5, 0.02**0.5, T) ** 2
    else:
        return np.linspace(1e-4, 0.02, T)

def get_alpha_bar(betas):
    alphas    = 1 - betas
    alpha_bar = np.cumprod(alphas)
    return alpha_bar

def forward_diffusion(x0, t, alpha_bar):
    """Add noise to x0 at timestep t."""
    sqrt_ab   = np.sqrt(alpha_bar[t])
    sqrt_1mab = np.sqrt(1 - alpha_bar[t])
    noise     = np.random.randn(*x0.shape)
    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise

def simulate_image_noising(
    image_size=32, T=1000,
    schedule='linear'
):
    np.random.seed(42)
    # Simulate a simple image pattern
    x, y = np.meshgrid(
        np.linspace(-2, 2, image_size),
        np.linspace(-2, 2, image_size))
    x0 = np.sin(x) * np.cos(y)
    x0 = (x0 - x0.min()) / \
         (x0.max() - x0.min())

    betas     = get_beta_schedule(T, schedule)
    alpha_bar = get_alpha_bar(betas)

    timesteps = [0, T//10, T//4,
                 T//2, 3*T//4, T-1]
    noisy_imgs = []

    for t in timesteps:
        xt, _ = forward_diffusion(
            x0, t, alpha_bar)
        noisy_imgs.append(xt)

    return x0, noisy_imgs, timesteps, \
           betas, alpha_bar

def simulate_denoising(
    image_size=32, T=100,
    schedule='linear', steps=10
):
    np.random.seed(42)
    betas     = get_beta_schedule(T, schedule)
    alpha_bar = get_alpha_bar(betas)

    # Start from pure noise
    xt = np.random.randn(
        image_size, image_size)

    # Target pattern
    x, y = np.meshgrid(
        np.linspace(-2, 2, image_size),
        np.linspace(-2, 2, image_size))
    target = np.sin(x) * np.cos(y)
    target = (target - target.min()) / \
             (target.max() - target.min())

    denoised = [xt.copy()]
    step_ids  = [T]

    for i in range(steps, 0, -1):
        t = int(i * T / steps) - 1
        t = max(0, min(T-1, t))

        # Simulate denoising step
        progress  = 1 - i/steps
        noise_level = np.sqrt(
            1 - alpha_bar[t])
        xt = (np.sqrt(alpha_bar[t]) *
               target +
               noise_level *
               np.random.randn(
                   image_size, image_size) *
               (1 - progress))

        xt = gaussian_filter(xt, sigma=0.3)
        denoised.append(xt.copy())
        step_ids.append(t)

    denoised.reverse()
    step_ids.reverse()
    return denoised, step_ids

def simulate_score_matching(
    n_points=200, noise_level=0.5
):
    np.random.seed(42)
    # 2D data distribution (two moons)
    n_half = n_points // 2
    theta1 = np.linspace(0, np.pi, n_half)
    theta2 = np.linspace(
        np.pi, 2*np.pi, n_half)
    x1 = np.column_stack([
        np.cos(theta1) * 2,
        np.sin(theta1) * 2])
    x2 = np.column_stack([
        np.cos(theta2) * 2 + 1,
        np.sin(theta2) * 2 - 0.5])
    data = np.vstack([x1, x2])
    data += np.random.randn(
        *data.shape) * 0.2

    # Add noise
    noisy = data + np.random.randn(
        *data.shape) * noise_level

    # Score = gradient of log p(x)
    # Approximate: pointing from noisy to clean
    scores = (data - noisy) / noise_level**2

    return data, noisy, scores

def simulate_ddpm_vs_ddim(
    T=1000, ddim_steps=[10, 50, 100, 200]
):
    betas     = get_beta_schedule(T)
    alpha_bar = get_alpha_bar(betas)

    results = {}

    # DDPM: all T steps
    ddpm_quality = []
    for t in range(T):
        quality = 1 - alpha_bar[t] + \
                  np.random.normal(0, 0.005)
        ddpm_quality.append(
            max(0, min(1, quality)))
    results['DDPM (1000 steps)'] = {
        'steps': T,
        'quality': round(
            (1 - ddpm_quality[-1]) * 100, 1),
        'time': 1000,
        'fid': 3.2
    }

    # DDIM: fewer steps
    for s in ddim_steps:
        quality = 1 - alpha_bar[
            int(T * 0.9)] + \
                  np.random.normal(0, 0.01) + \
                  s * 0.0003
        results['DDIM (' + str(s) +
                ' steps)'] = {
            'steps': s,
            'quality': round(
                max(0, min(1, quality)) *
                100, 1),
            'time': s,
            'fid': max(3.5, 3.2 +
                        (200-s) * 0.02)
        }

    return results

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = \
    st.tabs([
        "🌫️ Forward Process",
        "✨ Reverse Process",
        "📐 Score Matching",
        "⚡ DDPM vs DDIM",
        "🏗️ Architectures",
        "📚 Diffusion Guide"
    ])

# Tab 1 — Forward Process
with tab1:
    st.markdown("### 🌫️ Forward Diffusion Process")
    st.markdown(
        "Gradually add Gaussian noise to "
        "data over T timesteps until it "
        "becomes pure noise.")

    col1, col2 = st.columns([1, 3])

    with col1:
        T_fwd      = st.slider(
            "Total timesteps (T):",
            100, 1000, 500, 100)
        img_size   = st.select_slider(
            "Image size:",
            [16, 24, 32], value=24)
        schedule   = st.selectbox(
            "Noise schedule:",
            ["linear", "cosine",
             "quadratic"])
        show_stats = st.checkbox(
            "Show noise statistics", True)

        st.markdown("---")
        st.markdown("#### 📐 Forward Formula")
        st.markdown("""
    q(x_t | x_0) = N(x_t;
      √ᾱ_t × x_0,
      (1-ᾱ_t) × I)
              Where:
        - **ᾱ_t** = cumulative product
          of (1 - β_t)
        - **β_t** = noise schedule
        - As t→T: x_t → N(0,I)
        """)

    with col2:
        x0, noisy_imgs, timesteps, \
            betas, alpha_bar = \
            simulate_image_noising(
                img_size, T_fwd, schedule)

        # Show noising progression
        n_show = len(timesteps)
        cols_fw = st.columns(n_show)

        for i, (img, t) in enumerate(
            zip(noisy_imgs, timesteps)
        ):
            with cols_fw[i]:
                noise_pct = round(
                    (1-alpha_bar[t])*100, 1)
                fig_fw = go.Figure(
                    go.Heatmap(
                        z=img,
                        colorscale='Greys',
                        showscale=False,
                        zmin=-2, zmax=2
                    ))
                fig_fw.update_layout(
                    title='t=' + str(t) +
                          '<br>' +
                          str(noise_pct) +
                          '% noise',
                    height=160,
                    margin=dict(
                        l=5, r=5,
                        t=40, b=5),
                    xaxis=dict(
                        showticklabels=False),
                    yaxis=dict(
                        showticklabels=False),
                    paper_bgcolor='#0d1117',
                    plot_bgcolor='#0d1117',
                    font=dict(
                        color='white', size=9)
                )
                st.plotly_chart(
                    fig_fw,
                    use_container_width=True)

    # Noise schedule comparison
    st.markdown("---")
    st.markdown(
        "#### 📈 Noise Schedule Comparison")
    T_vis  = 1000
    scheds = ['linear', 'cosine',
              'quadratic']
    colors = ['#3498db', '#2ecc71',
              '#e74c3c']

    col1, col2 = st.columns(2)

    with col1:
        fig_beta = go.Figure()
        for sched, color in zip(
            scheds, colors
        ):
            b = get_beta_schedule(
                T_vis, sched)
            fig_beta.add_trace(go.Scatter(
                y=b,
                mode='lines',
                name=sched.capitalize(),
                line=dict(
                    color=color, width=2)
            ))
        fig_beta.update_layout(
            title='β_t (Noise Schedule)',
            xaxis_title='Timestep t',
            yaxis_title='β_t',
            height=300,
            template='plotly_dark'
        )
        st.plotly_chart(
            fig_beta,
            use_container_width=True)

    with col2:
        fig_ab = go.Figure()
        for sched, color in zip(
            scheds, colors
        ):
            b  = get_beta_schedule(
                T_vis, sched)
            ab = get_alpha_bar(b)
            fig_ab.add_trace(go.Scatter(
                y=ab,
                mode='lines',
                name=sched.capitalize(),
                line=dict(
                    color=color, width=2)
            ))
        fig_ab.add_hline(
            y=0.01,
            line_dash='dash',
            line_color='#f39c12',
            annotation_text=
                'Almost pure noise')
        fig_ab.update_layout(
            title='ᾱ_t (Signal Retention)',
            xaxis_title='Timestep t',
            yaxis_title='ᾱ_t',
            height=300,
            template='plotly_dark'
        )
        st.plotly_chart(
            fig_ab,
            use_container_width=True)

    if show_stats:
        betas_vis = get_beta_schedule(
            T_fwd, schedule)
        alpha_bar_vis = get_alpha_bar(
            betas_vis)
        st.markdown("#### 📊 Schedule Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("β_start",
                  round(betas_vis[0], 5))
        c2.metric("β_end",
                  round(betas_vis[-1], 4))
        c3.metric("ᾱ at T/2",
                  round(alpha_bar_vis[
                      T_fwd//2], 4))
        c4.metric("ᾱ at T",
                  round(alpha_bar_vis[-1],
                        6))

# Tab 2 — Reverse Process
with tab2:
    st.markdown("### ✨ Reverse Denoising Process")
    st.markdown(
        "The model learns to remove noise "
        "step by step, recovering the "
        "original data from pure noise.")

    col1, col2 = st.columns([1, 3])

    with col1:
        T_rev   = st.slider(
            "Total timesteps:", 50, 200,
            100, 10, key="T_rev")
        n_steps = st.slider(
            "Denoising steps to show:",
            5, 15, 8)
        sched_r = st.selectbox(
            "Schedule:",
            ["linear", "cosine"],
            key="sched_rev")
        img_r   = st.select_slider(
            "Image size:",
            [16, 24, 32], value=24,
            key="img_rev")

        if st.button(
            "✨ Run Denoising",
            type="primary",
            use_container_width=True
        ):
            denoised, step_ids = \
                simulate_denoising(
                    img_r, T_rev,
                    sched_r, n_steps)
            st.session_state[
                'denoised'] = denoised
            st.session_state[
                'step_ids'] = step_ids

        st.markdown("---")
        st.markdown("#### 📐 Reverse Formula")
        st.markdown("""
    p_θ(x_{t-1}|x_t) = N(x_{t-1};
      μ_θ(x_t, t),
      σ_t² × I)

        The UNet predicts noise ε_θ(x_t, t)

        Then:
            μ_θ = 1/√α_t × (x_t -
      β_t/√(1-ᾱ_t) × ε_θ)

        Trained with:
            L = E[||ε - ε_θ(x_t, t)||²]
                    """)

    with col2:
        if 'denoised' in st.session_state:
            imgs     = st.session_state[
                'denoised']
            step_ids = st.session_state[
                'step_ids']

            n_show = min(len(imgs), 8)
            cols_r = st.columns(n_show)
            step_subset = np.linspace(
                0, len(imgs)-1,
                n_show, dtype=int)

            for i, idx in enumerate(
                step_subset
            ):
                with cols_r[i]:
                    img = imgs[idx]
                    t   = step_ids[idx]
                    fig_rv = go.Figure(
                        go.Heatmap(
                            z=img,
                            colorscale='Viridis',
                            showscale=False
                        ))
                    label = "Noise" \
                        if i == 0 \
                        else ("Clean"
                              if i ==
                              n_show-1
                              else "t=" +
                                   str(t))
                    fig_rv.update_layout(
                        title=label,
                        height=160,
                        margin=dict(
                            l=5, r=5,
                            t=30, b=5),
                        xaxis=dict(
                            showticklabels=
                                False),
                        yaxis=dict(
                            showticklabels=
                                False),
                        paper_bgcolor=
                            '#0d1117',
                        plot_bgcolor=
                            '#0d1117',
                        font=dict(
                            color='white',
                            size=9)
                    )
                    st.plotly_chart(
                        fig_rv,
                        use_container_width=True)

            # Denoising quality over steps
            quality = [
                np.std(img)
                for img in imgs]
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                y=quality,
                mode='lines+markers',
                line=dict(
                    color='#2ecc71',
                    width=2),
                name='Signal Std Dev'
            ))
            fig_q.update_layout(
                title='Signal Recovery '
                      'During Denoising',
                xaxis_title=
                    'Denoising Step →',
                yaxis_title='Std Dev',
                height=250,
                template='plotly_dark'
            )
            st.plotly_chart(
                fig_q,
                use_container_width=True)
        else:
            st.info(
                "Click 'Run Denoising' "
                "to see the process!")

    # UNet architecture
    st.markdown("---")
    st.markdown(
        "#### 🏗️ UNet — The Noise Predictor")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        The **UNet** takes noisy image x_t
        and timestep t, predicts noise ε.

        **Key features:**
        - **Encoder:** Downsamples via Conv
        - **Bottleneck:** Global context
        - **Decoder:** Upsamples via Transpose Conv
        - **Skip connections:** Preserve detail
        - **Time embedding:** Sinusoidal
          embedding of timestep t
        - **Cross-attention:** For text
          conditioning (Stable Diffusion)
              Input: (x_t, t, [text prompt])
        ↓ Encoder (ResNet blocks)
        ↓ Attention (Self + Cross)
        ↓ Bottleneck
        ↓ Decoder (ResNet blocks)
        ↓ Attention
    Output: ε_θ (predicted noise)
            """)

    with col2:
        unet_layers = [
            ("Input x_t + time embed",
             "#3498db"),
            ("Conv + ResBlock (64)",
             "#2ecc71"),
            ("Downsample + Attention",
             "#2ecc71"),
            ("Conv + ResBlock (128)",
             "#2ecc71"),
            ("Downsample + Attention",
             "#2ecc71"),
            ("Bottleneck (256)",
             "#e74c3c"),
            ("Upsample + Skip",
             "#f39c12"),
            ("Conv + ResBlock (128)",
             "#f39c12"),
            ("Upsample + Skip",
             "#f39c12"),
            ("Conv + ResBlock (64)",
             "#f39c12"),
            ("Output: Predicted Noise",
             "#9b59b6")
        ]

        fig_unet = go.Figure()
        n_l = len(unet_layers)
        for i, (name, color) in \
                enumerate(unet_layers):
            y = (n_l - i) * 1.1
            fig_unet.add_shape(
                type='rect',
                x0=0.1, y0=y-0.4,
                x1=4.9, y1=y+0.4,
                fillcolor=color,
                line=dict(
                    color='white',
                    width=1),
                opacity=0.8)
            fig_unet.add_annotation(
                x=2.5, y=y,
                text=name,
                showarrow=False,
                font=dict(
                    color='white',
                    size=9))

        fig_unet.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            height=450,
            margin=dict(
                l=10, r=10,
                t=10, b=10)
        )
        st.plotly_chart(
            fig_unet,
            use_container_width=True)

# Tab 3 — Score Matching
with tab3:
    st.markdown("### 📐 Score Matching")
    st.markdown(
        "Score = gradient of log probability. "
        "The model learns the score function "
        "∇_x log p(x) to guide denoising.")

    col1, col2 = st.columns([1, 2])

    with col1:
        n_pts     = st.slider(
            "Data points:", 50, 300, 150)
        noise_lvl = st.slider(
            "Noise level σ:",
            0.1, 2.0, 0.5, 0.1)
        show_grid = st.checkbox(
            "Show score field grid", True)

        st.markdown("---")
        st.markdown("#### 💡 Score Function")
        st.markdown("""
        The **score function** is:
            s(x) = ∇_x log p(x)
                    It points in the direction of
        increasing probability density.

        **Why useful?**
        - Doesn't require normalizing
          constant of p(x)
        - Points toward high-density regions
        - Denoising = following scores

        **Denoising Score Matching:**
            L = E[||s_θ(x̃,σ) -
          ∇_x̃ log p(x̃|x)||²]
                  Where x̃ = x + σε is noisy data.

        The model learns to estimate
        scores at multiple noise levels.
        """)

    with col2:
        data_sm, noisy_sm, scores_sm = \
            simulate_score_matching(
                n_pts, noise_lvl)

        fig_sc = go.Figure()

        # Clean data
        fig_sc.add_trace(go.Scatter(
            x=data_sm[:, 0],
            y=data_sm[:, 1],
            mode='markers',
            name='Clean Data',
            marker=dict(
                color='#2ecc71',
                size=5, opacity=0.6)
        ))

        # Noisy data
        fig_sc.add_trace(go.Scatter(
            x=noisy_sm[:, 0],
            y=noisy_sm[:, 1],
            mode='markers',
            name='Noisy Data',
            marker=dict(
                color='#e74c3c',
                size=4, opacity=0.4)
        ))

        # Score arrows (subsample)
        if show_grid:
            step = max(1, n_pts // 30)
            for i in range(0, n_pts, step):
                scale = 0.3
                fig_sc.add_annotation(
                    x=noisy_sm[i, 0],
                    y=noisy_sm[i, 1],
                    ax=noisy_sm[i, 0] +
                       scores_sm[i, 0] *
                       scale,
                    ay=noisy_sm[i, 1] +
                       scores_sm[i, 1] *
                       scale,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#f39c12',
                    arrowwidth=1.5,
                    arrowsize=0.8
                )

        fig_sc.update_layout(
            title='Score Function — '
                  'Arrows Point to Clean Data',
            xaxis_title='x₁',
            yaxis_title='x₂',
            height=500,
            template='plotly_dark'
        )
        st.plotly_chart(
            fig_sc,
            use_container_width=True)

    # Score at different noise levels
    st.markdown("---")
    st.markdown(
        "#### 📊 Score Magnitude at "
        "Different Noise Levels")
    noise_levels = np.linspace(0.1, 3.0, 50)
    score_mags   = []
    for nl in noise_levels:
        _, noisy, scores = \
            simulate_score_matching(
                100, nl)
        score_mags.append(
            np.linalg.norm(
                scores, axis=1).mean())

    fig_sm = go.Figure()
    fig_sm.add_trace(go.Scatter(
        x=noise_levels, y=score_mags,
        mode='lines', fill='tozeroy',
        line=dict(
            color='#f39c12', width=2),
        fillcolor='rgba(243,156,18,0.15)',
        name='Score Magnitude'
    ))
    fig_sm.update_layout(
        title='Score Magnitude vs Noise Level',
        xaxis_title='Noise Level σ',
        yaxis_title='||∇ log p(x)||',
        height=300,
        template='plotly_dark'
    )
    st.plotly_chart(fig_sm,
                    use_container_width=True)

# Tab 4 — DDPM vs DDIM
with tab4:
    st.markdown("### ⚡ DDPM vs DDIM")
    st.markdown(
        "DDIM dramatically reduces inference "
        "steps while maintaining quality.")

    results_ddim = simulate_ddpm_vs_ddim()

    # Comparison table
    res_df = pd.DataFrame([
        {
            'Sampler':   name,
            'Steps':     v['steps'],
            'Rel Speed': str(round(
                1000 / v['steps'], 1)) + '×',
            'FID Score': v['fid'],
            'Quality':   str(v['quality']) +
                         '%'
        }
        for name, v in results_ddim.items()
    ])

    st.dataframe(
        res_df,
        use_container_width=True,
        hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        # Steps vs quality
        fig_steps = go.Figure()
        names  = list(results_ddim.keys())
        steps  = [results_ddim[n]['steps']
                  for n in names]
        fids   = [results_ddim[n]['fid']
                  for n in names]
        colors = ['#e74c3c'] + \
                 ['#3498db'] * (len(names)-1)

        fig_steps.add_trace(go.Bar(
            x=names, y=fids,
            marker_color=colors,
            text=[str(f) for f in fids],
            textposition='outside'
        ))
        fig_steps.add_hline(
            y=5.0, line_dash='dash',
            line_color='#2ecc71',
            annotation_text='Good FID < 5')
        fig_steps.update_layout(
            title='FID Score by Sampler '
                  '(Lower = Better)',
            yaxis_title='FID',
            height=350,
            template='plotly_dark',
            xaxis_tickangle=-20)
        st.plotly_chart(
            fig_steps,
            use_container_width=True)

    with col2:
        # Speed vs FID tradeoff
        fig_trade = go.Figure()
        fig_trade.add_trace(go.Scatter(
            x=steps, y=fids,
            mode='markers+text',
            text=names,
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(
                size=[15 if i == 0
                      else 10
                      for i in
                      range(len(names))],
                color=['#e74c3c'] +
                       ['#3498db'] *
                       (len(names)-1),
                opacity=0.85)
        ))
        fig_trade.update_layout(
            title='Steps vs FID Tradeoff',
            xaxis_title='Inference Steps',
            yaxis_title='FID Score',
            height=350,
            template='plotly_dark',
            xaxis_type='log'
        )
        st.plotly_chart(
            fig_trade,
            use_container_width=True)

    # DDPM vs DDIM explanation
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🐌 DDPM")
        st.markdown("""
        **Denoising Diffusion Probabilistic
        Models** (Ho et al. 2020)

        - Markovian process — each step
          depends on previous
        - Requires T=1000 steps
        - Stochastic denoising
        - ~30 seconds per image (GPU)

        **Sampling:**
            x_{t-1} = μ_θ(x_t,t) + σ_t × z
    where z ~ N(0, I)
            """)

    with col2:
        st.markdown("#### ⚡ DDIM")
        st.markdown("""
        **Denoising Diffusion Implicit
        Models** (Song et al. 2020)

        - Non-Markovian — can skip steps
        - 10-200 steps sufficient
        - Deterministic denoising
        - 10-100× faster than DDPM
        - Same quality at 50+ steps

        **Sampling:**
            x_{t-1} = √ᾱ_{t-1} × pred_x0
            + √(1-ᾱ_{t-1}) × ε_θ
                    No randomness → deterministic!
        """)

    # Sampler comparison
    st.markdown("---")
    st.markdown(
        "#### 🏎️ Modern Sampler Landscape")
    samplers_df = pd.DataFrame({
        'Sampler': [
            'DDPM', 'DDIM',
            'PNDM', 'DPM-Solver',
            'DPM-Solver++', 'UniPC',
            'LMS', 'Euler'
        ],
        'Min Steps': [
            1000, 20, 20, 10,
            10, 5, 15, 15],
        'Quality': [
            '✅✅', '✅✅', '✅✅',
            '✅✅', '✅✅✅', '✅✅✅',
            '✅✅', '✅✅'],
        'Deterministic': [
            '❌', '✅', '❌', '✅',
            '✅', '✅', '❌', '❌'],
        'Used In': [
            'Original paper',
            'Most SD models',
            'Early SD',
            'SD 2.0',
            'SD XL default',
            'Fast inference',
            'SD 1.x',
            'ComfyUI default']
    })
    st.dataframe(samplers_df,
                 use_container_width=True,
                 hide_index=True)

# Tab 5 — Architectures
with tab5:
    st.markdown(
        "### 🏗️ Diffusion Model Architectures")

    arch_choice = st.selectbox(
        "Architecture:",
        ["DDPM", "Stable Diffusion",
         "DALL-E 2", "Imagen",
         "Stable Diffusion XL"])

    arch_info = {
        "DDPM": {
            "desc":
                "Original pixel-space diffusion. "
                "Noises directly in pixel space.",
            "components": [
                "Noisy Image x_t (pixel space)",
                "UNet Noise Predictor",
                "Time Embedding (sinusoidal)",
                "Denoised Image x_0"
            ],
            "params": "86M",
            "resolution": "256×256",
            "condition": "Unconditional",
            "speed": "Slow (1000 steps)",
            "color": "#3498db"
        },
        "Stable Diffusion": {
            "desc":
                "Latent diffusion — works in "
                "compressed latent space for "
                "efficiency.",
            "components": [
                "Text Prompt",
                "CLIP Text Encoder",
                "Latent Noise z_t",
                "UNet (in latent space)",
                "Cross-Attention (text→image)",
                "VAE Decoder",
                "Output Image"
            ],
            "params": "860M",
            "resolution": "512×512",
            "condition": "Text (CLIP)",
            "speed": "~5s (20 steps)",
            "color": "#9b59b6"
        },
        "DALL-E 2": {
            "desc":
                "CLIP-guided diffusion with "
                "prior network mapping text "
                "to image embeddings.",
            "components": [
                "Text Prompt",
                "CLIP Text Encoder",
                "Prior Network (text→CLIP image embed)",
                "Diffusion Decoder",
                "Super-Resolution Cascade",
                "1024×1024 Output"
            ],
            "params": "3.5B",
            "resolution": "1024×1024",
            "condition": "Text + CLIP prior",
            "speed": "~10s",
            "color": "#e74c3c"
        },
        "Imagen": {
            "desc":
                "Google's cascaded diffusion "
                "with frozen T5 text encoder.",
            "components": [
                "Text Prompt",
                "T5-XXL Text Encoder (frozen)",
                "Base 64×64 Diffusion Model",
                "Super-Res: 64→256",
                "Super-Res: 256→1024",
                "1024×1024 Output"
            ],
            "params": "2B",
            "resolution": "1024×1024",
            "condition": "Text (T5)",
            "speed": "~15s",
            "color": "#2ecc71"
        },
        "Stable Diffusion XL": {
            "desc":
                "Larger SD with two-stage "
                "pipeline and improved "
                "composition.",
            "components": [
                "Text Prompt",
                "CLIP-L + OpenCLIP encoders",
                "Base Model UNet (2.6B)",
                "Latent Space 128×128",
                "Refiner Model UNet",
                "VAE Decoder",
                "1024×1024 Output"
            ],
            "params": "6.6B",
            "resolution": "1024×1024",
            "condition": "Text (dual CLIP)",
            "speed": "~8s",
            "color": "#f39c12"
        }
    }

    info = arch_info[arch_choice]
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            "<div style='background:#1a1a2e;"
            "border:2px solid " +
            info['color'] + ";"
            "border-radius:12px;"
            "padding:20px'>"
            "<h3 style='color:" +
            info['color'] +
            ";margin:0'>" +
            arch_choice + "</h3>"
            "<p style='color:#8b949e;"
            "margin:8px 0'>" +
            info['desc'] + "</p>"
            "</div>",
            unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        c1.metric("Parameters",
                  info['params'])
        c2.metric("Resolution",
                  info['resolution'])
        c1.metric("Conditioning",
                  info['condition'])
        c2.metric("Speed",
                  info['speed'])

    with col2:
        # Architecture flow
        components = info['components']
        n_c        = len(components)
        fig_ac     = go.Figure()

        for i, comp in enumerate(components):
            y = (n_c - i) * 1.2
            fig_ac.add_shape(
                type='rect',
                x0=0.1, y0=y-0.4,
                x1=4.9, y1=y+0.4,
                fillcolor=info['color'],
                line=dict(
                    color='white', width=1),
                opacity=0.75)
            fig_ac.add_annotation(
                x=2.5, y=y,
                text=comp,
                showarrow=False,
                font=dict(
                    color='white', size=10))
            if i < n_c - 1:
                fig_ac.add_annotation(
                    x=2.5, y=y-0.4,
                    ax=2.5, ay=y-0.8,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='white',
                    arrowwidth=2)

        fig_ac.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False),
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            height=500,
            margin=dict(
                l=10, r=10,
                t=10, b=10)
        )
        st.plotly_chart(
            fig_ac,
            use_container_width=True)

    # Model comparison
    st.markdown("---")
    st.markdown(
        "#### 📊 Diffusion Model Timeline")
    timeline = pd.DataFrame({
        'Year': [2020, 2021, 2021,
                 2022, 2022, 2022,
                 2023, 2023],
        'Model': [
            'DDPM', 'DDIM', 'DALL-E',
            'Stable Diffusion 1.x',
            'DALL-E 2', 'Imagen',
            'SD XL', 'DALL-E 3'],
        'FID': [3.2, 4.0, None,
                8.5, 4.5, 7.3,
                3.8, None],
        'Key Innovation': [
            'Probabilistic diffusion',
            'Non-Markovian fast sampling',
            'Text-to-image at scale',
            'Latent diffusion (open source)',
            'CLIP prior network',
            'Cascaded + T5 text encoder',
            'Dual CLIP + larger model',
            'Improved prompt following']
    })
    st.dataframe(timeline,
                 use_container_width=True,
                 hide_index=True)

# Tab 6 — Guide
with tab6:
    st.markdown("### 📚 Diffusion Models Guide")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 🌊 Core Idea

        **Forward process** q:
        Gradually corrupt data with noise
        over T steps. At step T: pure noise.

        **Reverse process** p_θ:
        Train a neural network to undo
        the noise, one step at a time.

        **Key insight:**
        If we can reverse a small noising step,
        we can generate data by reversing
        the entire process starting from noise.

        #### 🧮 Training Objective

        Simple noise prediction loss:
            L_simple = E_{t,x_0,ε}[
      ||ε - ε_θ(√ᾱ_t x_0 +
                 √(1-ᾱ_t)ε, t)||²
    ]

        Sample t ~ Uniform(1, T)
        Sample ε ~ N(0, I)
        Compute x_t using forward formula
        Predict ε from x_t using UNet
        Minimize MSE between true and
        predicted noise.

        #### 🆚 Diffusion vs GANs

        | | GANs | Diffusion |
        |---|---|---|
        | Training | Unstable | Stable |
        | Mode collapse | Yes | No |
        | Quality | Good | Better |
        | Speed | Fast | Slow |
        | Diversity | Low | High |
        | Controllable | Hard | Easy |
        """)

    with col2:
        st.markdown("""
        #### 🏗️ Latent Diffusion (SD)

        **Problem with pixel space:**
        High-res images = huge computation.
        1024×1024 = 3M pixels per step.

        **Latent space solution:**
        1. VAE encoder compresses image
           to small latent (64×64×4)
        2. Diffusion in latent space
           (64× smaller!)
        3. VAE decoder expands back

        **Text conditioning:**
        CLIP encodes text to embedding.
        Cross-attention in UNet lets image
        features attend to text features.

        #### 🔮 Inference Pipeline (SD)
            1. Encode text → CLIP embedding
    2. Sample z_T ~ N(0, I) (latent)
    3. For t = T, T-1, ..., 1:
       ε = UNet(z_t, t, text_embed)
       z_{t-1} = DDIM_step(z_t, ε, t)
    4. Decode z_0 → image via VAE

        #### 🎛️ Key Parameters

        **CFG Scale (Classifier-Free Guidance):**
        How strictly to follow prompt.
        High CFG = more prompt adherent,
        less creative. Typical: 7-12.
            ε_guided = ε_uncond +
      cfg × (ε_cond - ε_uncond)

        **Steps:** 20-50 usually sufficient
        with DDIM. More = diminishing returns.

        **Seed:** Controls initial noise.
        Same seed + prompt = same image.
        """)

    st.markdown("---")
    st.markdown("#### 🎯 Interview Questions")
    qs = [
        ("What is the forward diffusion process?",
         "A Markov chain that gradually adds "
         "Gaussian noise to data over T steps. "
         "q(x_t|x_0) = N(x_t; √ᾱ_t x_0, "
         "(1-ᾱ_t)I). At t=T, data becomes "
         "isotropic Gaussian noise."),
        ("What does the UNet predict?",
         "The added noise ε at timestep t. "
         "Given noisy image x_t and t, it "
         "predicts the noise ε that was added. "
         "We can then recover a denoised estimate."),
        ("What is classifier-free guidance?",
         "A technique to improve sample quality "
         "and prompt adherence. Train UNet with "
         "and without conditioning. At inference, "
         "interpolate: ε = ε_uncond + s×(ε_cond - ε_uncond). "
         "Higher s = stronger conditioning."),
        ("Why use latent space?",
         "Pixel space is computationally "
         "expensive for high-res images. "
         "VAE compresses image to small latent "
         "(e.g. 64×64 instead of 512×512), "
         "making diffusion 64× cheaper while "
         "maintaining quality."),
        ("DDPM vs DDIM?",
         "DDPM is stochastic Markovian — needs "
         "all T steps. DDIM is deterministic "
         "non-Markovian — can skip steps, "
         "10-100× faster with similar quality. "
         "Same trained model, different sampler.")
    ]
    for q, a in qs:
        with st.expander("❓ " + q):
            st.markdown("**Answer:** " + a)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Diffusion Model Visualizer | "
    "Phase 4: Deep Learning · DDPM · DDIM · "
    "Stable Diffusion"
)