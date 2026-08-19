let express = require("express");
let app = express();
let http = require("http");
let server = http.createServer(app);
let fs = require("fs");

app.use(express.json());
app.use("/css", express.static("css"));
app.use("/js", express.static("js"));
app.use(express.static(__dirname));

app.get("/", function(req, res) {
    res.sendFile(__dirname + "/index.html");
});

server.listen(3001, '0.0.0.0', function() {
    console.log("Listening on port 3001");
});

app.post("/api/convert", async (req, res) => {
    const input = req.body.text;
    console.log("Get Request:", input);

    const response = await fetch("http://localhost:8001/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: input})
    });
    data = await response.json()

    console.log("Get Response:", data.result);
    res.json({result: data.result});
});
