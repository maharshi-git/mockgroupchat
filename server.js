/**
 * Group Chat API Server
 *
 * Endpoints:
 *   POST /api/run     — Start a new group chat (returns runId)
 *   POST /api/resume  — Resume a paused chat with user input (uses runId)
 *   GET  /api/flows   — List available flows
 */

const express = require("express");
const { startGroupChat, resumeGroupChat } = require("./groupchat/groupchat");

const app = express();
app.use(express.json());

// ───────────────────────────────────────────────────
//  POST /api/run — Start a new group chat
//  Body: { flowId: string, message: string }
// ───────────────────────────────────────────────────
app.post("/api/run", async (req, res) => {
    try {
        const { flowId, message } = req.body;
        if (!flowId || !message) {
            return res
                .status(400)
                .json({ error: "flowId and message are required" });
        }

        const result = await startGroupChat(flowId, message);
        res.json(result);
    } catch (err) {
        console.error("[Server Error] POST /api/run:", err.message);
        res.status(500).json({ error: err.message });
    }
});

// ───────────────────────────────────────────────────
//  POST /api/resume — Resume a paused group chat
//  Body: { flowId: string, runId: string, userInput: string }
//
//  The runId is the same one returned from /api/run
//  when status was "interrupted". The MemorySaver
//  checkpointer restores state for that thread_id.
// ───────────────────────────────────────────────────
app.post("/api/resume", async (req, res) => {
    try {
        const { flowId, runId, userInput } = req.body;
        if (!flowId || !runId || !userInput) {
            return res.status(400).json({
                error: "flowId, runId, and userInput are required",
            });
        }

        const result = await resumeGroupChat(flowId, runId, userInput);
        res.json(result);
    } catch (err) {
        console.error("[Server Error] POST /api/resume:", err.message);
        res.status(500).json({ error: err.message });
    }
});

// ───────────────────────────────────────────────────
//  GET /api/flows — List all available flows
// ───────────────────────────────────────────────────
app.get("/api/flows", (req, res) => {
    try {
        const flows = require("./groupchat/flows.json");
        res.json({
            flows: flows.flows.map((f) => ({
                flowID: f.flowID,
                flowName: f.flowName,
                flowDescription: f.flowDescription,
                agents:
                    f.nodes.find((n) => n.type === "groupchat")?.agents || [],
                endCondition:
                    f.nodes.find((n) => n.type === "groupchat")?.endCondition ||
                    "",
            })),
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ───────────────────────────────────────────────────
//  Start
// ───────────────────────────────────────────────────
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`\n🚀 Group Chat Server running on http://localhost:${PORT}`);
    console.log(`\n📋 Endpoints:`);
    console.log(`   POST /api/run       Start a new group chat`);
    console.log(`                       Body: { flowId, message }`);
    console.log(`   POST /api/resume    Resume a paused chat`);
    console.log(`                       Body: { flowId, runId, userInput }`);
    console.log(`   GET  /api/flows     List available flows\n`);
});
