from dash import dcc, html, Input, Output, callback

layout = html.Div([
    html.H3('Page onnx'),

"""

    <script src="script_onnx.js" type="module"></script>
  </head>
  <body>
    <div class="navbar">
      <ul>
        <li>
          <a class="active" href="/index.php"
            ><i class="fa fa-fw fa-home"></i> Home</a
          >
        </li>
        <li>
          <a href="/ressources/html/training.php"
            ><i class="fas fa-share-alt"></i> Training</a
          >
        </li>
        <li>
          <a href="/ressources/html/interface.php"
            ><i class="fas fa-photo-video"></i> Inference</a
          >
        </li>
        <li>
          <a href="/ressources/onnx/index_onnx.php"
            ><i class="	fas fa-pen-alt"></i> ONNX</a
          >
        </li>
        <li>
          <a href="/ressources/html/doc.php"
            ><i class="fas fa-book"></i> Documentation</a
          >
        </li>
      </ul>
    </div>

    <div class="drawing_zone">
      <canvas
        class="canvas elevation"
        id="canvas"
        width="280"
        height="280"
      ></canvas>

      <div class="button" id="clear-button">CLEAR</div>

      <div class="predictions">
        <div class="prediction-col" id="prediction-0">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">0</div>
        </div>

        <div class="prediction-col" id="prediction-1">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">1</div>
        </div>

        <div class="prediction-col" id="prediction-2">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">2</div>
        </div>

        <div class="prediction-col" id="prediction-3">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">3</div>
        </div>

        <div class="prediction-col" id="prediction-4">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">4</div>
        </div>

        <div class="prediction-col" id="prediction-5">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">5</div>
        </div>

        <div class="prediction-col" id="prediction-6">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">6</div>
        </div>

        <div class="prediction-col" id="prediction-7">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">7</div>
        </div>

        <div class="prediction-col" id="prediction-8">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">8</div>
        </div>

        <div class="prediction-col" id="prediction-9">
          <div class="prediction-bar-container">
            <div class="prediction-bar"></div>
          </div>
          <div class="prediction-number">9</div>
        </div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>
  </body>
</html>
"""

])

