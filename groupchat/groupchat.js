/**
 * Group Chat Engine — LangGraph Multi-Agent Orchestration
 *
 * Architecture:
 *   START → supervisor → [agentX | FINISH→END]
 *   agentX → (notificationSent?) → waitForInput | endConditionCheck
 *   waitForInput (interrupt) → endConditionCheck
 *   endConditionCheck → [supervisor | END]
 *
 * - MemorySaver checkpointer persists state per runId (thread_id)
 * - send_notification tool triggers interrupt() → pauses graph
 * - External user resumes via POST /api/resume with runId + userInput
 * - End condition agent evaluates after every agent round
 *
 * NOTE: Uses prompt-based tool calling (LLM outputs JSON) since
 *       phi3:mini does not support native Ollama tool calling.
 */

const {
    Annotation,
    StateGraph,
    START,
    END,
    MemorySaver,
    Command,
    interrupt,
    messagesStateReducer,
} = require("@langchain/langgraph");
const { ChatOllama } = require("@langchain/ollama");
const {
    HumanMessage,
    AIMessage,
    SystemMessage,
} = require("@langchain/core/messages");
const fs = require("fs");
const path = require("path");

// ═══════════════════════════════════════════════════════════
//  Configuration — loaded once at module init
// ═══════════════════════════════════════════════════════════

const agentsConfig = JSON.parse(
    fs.readFileSync(path.join(__dirname, "agents.json"), "utf8")
);
const flowsConfig = JSON.parse(
    fs.readFileSync(path.join(__dirname, "flows.json"), "utf8")
);
const servicesConfig = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "services.json"), "utf8")
);

// ═══════════════════════════════════════════════════════════
//  Singletons — shared across all runs
// ═══════════════════════════════════════════════════════════

/** MemorySaver keeps state in-memory, keyed by thread_id (runId) */
const checkpointer = new MemorySaver();

/** Local Ollama LLM — used by all agents and the supervisor */
const llm = new ChatOllama({ model: "phi3:mini", temperature: 0 });

/** Cache compiled graphs per flowId to avoid recompilation */
const graphCache = new Map();

/** Max conversation rounds before forced termination */
const MAX_ROUNDS = 15;

// ═══════════════════════════════════════════════════════════
//  State Annotation
// ═══════════════════════════════════════════════════════════

const GroupChatState = Annotation.Root({
    /** Full conversation history — uses LangGraph's message dedup reducer */
    messages: Annotation({
        reducer: messagesStateReducer,
        default: () => [],
    }),
    /** Which agent the supervisor picked for the current round */
    nextAgent: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => "",
    }),
    /** Set to true by endConditionCheck when the goal is achieved */
    isComplete: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => false,
    }),
    /** The goal description from flows.json endCondition */
    endCondition: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => "",
    }),
    /** Tracks how many supervisor→agent rounds have executed */
    roundCount: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => 0,
    }),
    /** Flag set by agent nodes when send_notification tool is called */
    notificationSent: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => false,
    }),
    /** High-level guidance from flows.json flowDescription */
    flowDirection: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => "",
    }),
    /** Detailed orchestration status (complete, PendingWaitingForInput, etc.) */
    detailedStatus: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => "PendingLookingForSolution",
    }),
    /** Current run status: 'running', 'completed', 'interrupted', or 'error' */
    status: Annotation({
        reducer: (x, y) => (y !== undefined ? y : x),
        default: () => "running",
    }),
});

// ═══════════════════════════════════════════════════════════
//  API Pipeline Executor
//  Walks graphNodes/graphEdges from agents.json to make
//  real HTTP calls through the mock server.
// ═══════════════════════════════════════════════════════════

async function executeGraphPipeline(graphNodes, graphEdges, inputPayload) {
    if (!graphNodes?.length) {
        return { status: "ok", message: "No API pipeline defined" };
    }

    // Build lookup and find the start node (node with no incoming edges)
    const nodeMap = Object.fromEntries(graphNodes.map((n) => [n.id, n]));
    const incomingSet = new Set(graphEdges.map((e) => e.to));
    const startNode = graphNodes.find((n) => !incomingSet.has(n.id));
    if (!startNode) return { error: "No start node found in pipeline" };

    let currentId = startNode.id;
    let payload = inputPayload;
    let result = { status: "ok" };

    while (currentId) {
        const node = nodeMap[currentId];
        if (!node) break;

        console.log(`    [Pipeline] Executing node: ${node.id} (${node.type})`);

        switch (node.type) {
            case "insertPayload":
                payload = inputPayload;
                console.log(`    [Pipeline] Injected payload:`, JSON.stringify(payload));
                break;

            case "apiCall": {
                const baseUrl =
                    servicesConfig[node.serviceName] || "http://localhost:3001";
                const url = `${baseUrl}/${node.entitySet}`;

                console.log(`    [Pipeline] Calling API: ${node.crudType} ${url}`);

                try {
                    if (
                        node.crudType === "CREATE" ||
                        node.crudType === "UPDATE"
                    ) {
                        const resp = await fetch(url, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(payload),
                        });
                        result = await resp.json();
                    } else {
                        // READ → GET with query params
                        const params = new URLSearchParams();
                        if (payload && typeof payload === "object") {
                            for (const [k, v] of Object.entries(payload)) {
                                if (v != null && v !== "")
                                    params.set(k, String(v));
                            }
                        }
                        const readUrl = params.toString()
                            ? `${url}?${params}`
                            : url;
                        const resp = await fetch(readUrl);
                        result = await resp.json();
                    }
                    console.log(`    [Pipeline] API Result:`, JSON.stringify(result).substring(0, 100));
                } catch (err) {
                    console.error(`    [Pipeline API Error] ${url}:`, err.message);
                    result = { error: `API call failed: ${err.message}`, url };
                }
                payload = result;
                break;
            }

            case "returnObject":
                result = payload;
                console.log(`    [Pipeline] Return object:`, JSON.stringify(result).substring(0, 100));
                break;
        }

        // Follow edge to next node in the pipeline
        const edge = graphEdges.find((e) => e.from === currentId);
        currentId = edge ? edge.to : null;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════
//  Prompt-Based Tool Descriptions
//  Since phi3:mini doesn't support native tool calling,
//  we describe tools in the system prompt and parse
//  JSON output from the LLM to determine which tool to run.
// ═══════════════════════════════════════════════════════════

/**
 * Build a text description of an agent's tools for the system prompt.
 * Returns { toolDescriptions: string, toolLookup: { [name]: toolConfig } }
 */
function describeToolsForPrompt(agentConfig) {
    const toolLookup = {};
    const descriptions = [];

    for (const tc of agentConfig.tools || []) {
        if (!tc.toolName) continue;
        toolLookup[tc.toolName] = tc;

        // Build field descriptions
        const fields = [];
        if (tc.sampleForm?.length) {
            for (const f of tc.sampleForm) {
                fields.push(
                    `    "${f.label}": "${f.fieldLabel || f.label}"${f.mandatory ? " (REQUIRED)" : " (optional)"}`
                );
            }
        } else if (tc.formToBeSent) {
            for (const [key, val] of Object.entries(tc.formToBeSent)) {
                if (typeof val !== "object" || val === null) {
                    fields.push(`    "${key}": "${typeof val}"`);
                }
            }
        }

        descriptions.push(
            `  TOOL: ${tc.toolName}\n` +
            `  Description: ${tc.toolDefinition || "No description"}\n` +
            `  Input fields:\n${fields.join("\n") || "    (none)"}`
        );
    }

    return {
        toolDescriptions: descriptions.join("\n\n"),
        toolLookup,
    };
}

/**
 * Try to parse a tool call from the LLM's text response.
 * Expected format:
 *   { "toolCall": "tool_name", "args": { "field": "value" } }
 * Returns null if no tool call is detected.
 */
function parseToolCallFromResponse(text) {
    // Strip markdown code fences
    let cleaned = text.replace(/```(?:json)?\s*/gi, "").replace(/```/g, "").trim();

    // Find the start of a potential JSON object
    let startIdx = cleaned.indexOf('{');
    if (startIdx === -1) return null;

    // Extract potential JSON by counting braces to handle nesting
    let bracketCount = 0;
    let jsonStr = "";
    for (let i = startIdx; i < cleaned.length; i++) {
        const char = cleaned[i];
        if (char === '{') bracketCount++;
        else if (char === '}') bracketCount--;

        jsonStr += char;
        if (bracketCount === 0) break;
    }

    if (!jsonStr) return null;

    try {
        const parsed = JSON.parse(jsonStr);
        // Support multiple common keys
        const name = parsed.toolCall || parsed.tool || parsed.tool_name || parsed.toolCalled;
        if (name && typeof name === "string") {
            return {
                name: name,
                args: parsed.args || parsed.arguments || parsed.parameters || parsed.filledForm || {},
            };
        }
    } catch (e) {
        console.log(`    [Parser] Failed to parse extracted JSON:`, e.message);
        // Fallback: try regex for simpler objects if manual extraction failed
        const jsonMatches = cleaned.match(/\{[\s\S]*\}/g);
        if (jsonMatches) {
            for (const match of jsonMatches) {
                try {
                    const p = JSON.parse(match);
                    const n = p.toolCall || p.tool || p.tool_name || p.toolCalled;
                    if (n) return { name: n, args: p.args || p.arguments || {} };
                } catch { }
            }
        }
    }
    return null;
}

// ═══════════════════════════════════════════════════════════
//  Graph Builder
//  Constructs a LangGraph StateGraph for a given flowId.
//  The graph includes:
//    - A supervisor node (routes to agents or FINISH)
//    - One node per agent (with prompt-based tool calling)
//    - A waitForInput node (calls interrupt() on notification)
//    - An endConditionCheck node (evaluates goal completion)
// ═══════════════════════════════════════════════════════════

function buildGroupChatGraph(flowId) {
    // ---- Resolve flow & agents ----
    const flow = flowsConfig.flows.find((f) => f.flowID === flowId);
    if (!flow) throw new Error(`Flow '${flowId}' not found in flows.json`);

    const gcNode = flow.nodes.find((n) => n.type === "groupchat");
    if (!gcNode)
        throw new Error(`No groupchat node found in flow '${flowId}'`);

    const agentNames = gcNode.agents || [];
    const endCondition = gcNode.endCondition || "All tasks are completed";

    const activeAgents = agentNames
        .map((name) => agentsConfig.agents.find((a) => a.agentName === name))
        .filter(Boolean);

    if (activeAgents.length === 0) {
        throw new Error(
            `No matching agents in agents.json for flow. Expected: ${agentNames.join(", ")}`
        );
    }

    // Build tool descriptions per agent (prompt-based, not LLM-native)
    const agentToolInfoMap = {};
    for (const agent of activeAgents) {
        agentToolInfoMap[agent.agentName] = describeToolsForPrompt(agent);
    }

    // ---- StateGraph ----
    const graph = new StateGraph(GroupChatState);

    // ┌──────────────────────────────────────────────────────┐
    // │  NODE: supervisor                                     │
    // │  Picks which agent should act next, or FINISH.        │
    // └──────────────────────────────────────────────────────┘
    graph.addNode("supervisor", async (state) => {
        // --- 1. EVALUATION PHASE (The End Condition Agent) ---
        const evalPrompt = `You are an evaluator checking if a goal has been achieved in a multi-agent conversation.
GOAL: "${endCondition || "Task completion"}"

Review the conversation and determine if this goal has been FULLY achieved.
The goal is achieved ONLY when the actual operation completed successfully (e.g., an API returned a success response), not merely discussed or planned.

If achieved, respond with exactly "YES". Otherwise, respond with exactly "NO". No other text.`;

        const evalResponse = await llm.invoke([
            new SystemMessage(evalPrompt),
            ...state.messages,
        ]);

        const isGoalAchieved = evalResponse.content.trim().toUpperCase().startsWith("YES");

        if (isGoalAchieved) {
            console.log("  [Supervisor] Goal achieved! → complete");
            return { 
                nextAgent: "FINISH", 
                status: "completed", 
                detailedStatus: "complete",
                isComplete: true 
            };
        }

        // --- 2. ORCHESTRATION PHASE (The Orchestrator Agent) ---
        const agentDescriptions = activeAgents
            .map(
                (a) =>
                    `- ${a.agentName}: ${(a.agentDefinition || "General purpose agent").substring(0, 200)}`
            )
            .join("\n");

        const orchPrompt = `You are an Orchestrator managing a group chat to achieve a goal.
GOAL: ${endCondition}
FLOW DIRECTION: ${state.flowDirection}

AVAILABLE AGENTS:
${agentDescriptions}

### HALLUCINATION GUARD (CRITICAL):
1. **DO NOT INVENT TECHNICAL DATA.** Never make up wage type IDs, ServiceNow ticket numbers, or dates.
2. **VERIFY CONTEXT.** Only use technical values if they have been explicitly provided by the user in the conversation history.
3. **ASK IF MISSING.** If a critical ID or value is missing to proceed, you MUST respond with the USER format.

### DECISION RULES:
Based on the conversation, decide the ONE NEXT step. 
Output ONLY ONE of these two formats (no preamble, no continuation):
AGENT: <AgentName> | STATUS: <One of the 4 Statuses>
USER: <Specific Question for the Human> | STATUS: <One of the 4 Statuses>

Statuses: PendingWaitingForInput, PendingLookingForSolution, FailureLookingForInfo, FailureWaitForUserInput

### EXAMPLES (OUTPUT ONLY ONE LINE):
AGENT: WageTypeAgent | STATUS: PendingLookingForSolution
USER: Please provide the start date for wage type 1001. | STATUS: PendingWaitingForInput`;

        const orchResponse = await llm.invoke([
            new SystemMessage(orchPrompt),
            ...state.messages,
        ]);

        const content = orchResponse.content.trim().split('\n')[0]; // Strictly first line
        console.log(`  [Supervisor] Orchestrator decision: ${content}`);

        let selectedAgent = "FINISH";
        let detailedStatus = "PendingLookingForSolution";
        let techStatus = "running";
        let newMessages = [];

        if (content.toUpperCase().includes("USER:")) {
            selectedAgent = "waitForInput";
            techStatus = "interrupted";
            
            // Extract the question part
            const questionMatch = content.match(/USER: (.*?) \| STATUS:/i) || content.match(/USER: (.*)/i);
            const question = questionMatch ? questionMatch[1] : content;
            
            detailedStatus = content.toUpperCase().includes("FAILURE") ? "FailureWaitForUserInput" : "PendingWaitingForInput";
            
            // ADD THE QUESTION TO THE MESSAGE HISTORY SO USER SEES IT
            newMessages.push(new AIMessage(`[Question] ${question}`));
        } else {
            // Find which agent was named
            const sanitizedContent = content.toLowerCase().replace(/[^a-z0-9]/g, "");
            for (const name of agentNames) {
                const sanitizedName = name.toLowerCase().replace(/[^a-z0-9]/g, "");
                if (sanitizedContent.includes(sanitizedName)) {
                    selectedAgent = name;
                    break;
                }
            }
            detailedStatus = content.toUpperCase().includes("FAILURE") ? "FailureLookingForInfo" : "PendingLookingForSolution";
            techStatus = "running";

            // Add orchestrator guidance to history for keeping agents on track
            newMessages.push(new AIMessage(`[Orchestrator] Next step: ${content}`));
        }

        // Safety: enforce round limit
        const newRound = (state.roundCount || 0) + 1;
        if (newRound > MAX_ROUNDS) {
            console.log(`  [Supervisor] ⚠ Max rounds (${MAX_ROUNDS}) reached → FINISH`);
            selectedAgent = "FINISH";
            techStatus = "completed";
            detailedStatus = "complete";
            return { 
                nextAgent: "FINISH", 
                status: "completed", 
                detailedStatus: "complete", 
                roundCount: newRound 
            };
        }

        console.log(`  [Supervisor] Round ${newRound} → ${selectedAgent} (Status: ${detailedStatus})`);
        return { 
            nextAgent: selectedAgent, 
            status: techStatus, 
            detailedStatus, 
            roundCount: newRound,
            notificationSent: false, // reset
            messages: newMessages
        };
    });

    // ┌──────────────────────────────────────────────────────┐
    // │  NODES: one per agent                                 │
    // │  Uses PROMPT-BASED tool calling:                      │
    // │    - System prompt describes available tools           │
    // │    - LLM outputs JSON with toolCall + args            │
    // │    - We parse it, execute the API pipeline, and       │
    // │      feed the result back for a final answer          │
    // │  Sets notificationSent=true if send_notification used │
    // └──────────────────────────────────────────────────────┘
    for (const agent of activeAgents) {
        const { toolDescriptions, toolLookup } =
            agentToolInfoMap[agent.agentName];
        const hasTools = Object.keys(toolLookup).length > 0;

        graph.addNode(agent.agentName, async (state) => {
            console.log(`  [${agent.agentName}] Executing...`);

            let toolCallPrompt = "";
            if (hasTools) {
                toolCallPrompt = `

YOUR AVAILABLE TOOLS:
${toolDescriptions}

TOOL CALLING INSTRUCTIONS:
If you need to perform an action or fetch data, you MUST call one of the tools above.
To call a tool, respond with ONLY this JSON format (no other text if you are calling a tool):
{ "toolCall": "tool_name", "args": { "field": "value", ... } }

IMPORTANT: 
- Use the exact "toolCall" name from the list above.
- Provide all required arguments in the "args" object.
- Fill in the args with values from the conversation history.
- If you have already executed a tool and have the result, you can provide a text response to the user.`;
            }

            const sysMsg = new SystemMessage(
                `${agent.agentDefinition || "You are a helpful agent."}

You are part of a group chat working towards this goal: ${endCondition}
Contribute your expertise based on the conversation.
${toolCallPrompt}`
            );

            const MAX_TOOL_ITERS = 3;
            const newMessages = [];
            let notificationSent = false;

            for (let iter = 0; iter < MAX_TOOL_ITERS; iter++) {
                const allMsgs = [sysMsg, ...state.messages, ...newMessages];
                const response = await llm.invoke(allMsgs);
                const responseText = response.content || "";

                // Try to parse a tool call from the response
                const toolCall = hasTools
                    ? parseToolCallFromResponse(responseText)
                    : null;

                if (!toolCall) {
                    // No tool call — this is the agent's final response
                    newMessages.push(
                        new AIMessage({
                            content: responseText,
                            name: agent.agentName,
                        })
                    );
                    console.log(
                        `  [${agent.agentName}] Responded (no tool call)`
                    );
                    break;
                }

                // We have a tool call — execute it
                console.log(
                    `  [${agent.agentName}] Tool call: ${toolCall.name}(${JSON.stringify(toolCall.args)})`
                );

                const toolConfig = toolLookup[toolCall.name];
                if (!toolConfig) {
                    // Tool not found — tell the agent
                    newMessages.push(
                        new AIMessage({
                            content: responseText,
                            name: agent.agentName,
                        })
                    );
                    newMessages.push(
                        new HumanMessage(
                            `[System]: Tool '${toolCall.name}' not found. Available tools: ${Object.keys(toolLookup).join(", ")}`
                        )
                    );
                    continue;
                }

                // Execute the API pipeline for this tool
                let apiResult;
                try {
                    apiResult = await executeGraphPipeline(
                        toolConfig.graphNodes || [],
                        toolConfig.graphEdges || [],
                        toolCall.args
                    );
                } catch (err) {
                    apiResult = { error: err.message };
                }

                console.log(
                    `  [${agent.agentName}] Tool result: ${JSON.stringify(apiResult).substring(0, 200)}`
                );

                // Add the tool call and result as messages
                newMessages.push(
                    new AIMessage({
                        content: responseText,
                        name: agent.agentName,
                    })
                );
                newMessages.push(
                    new HumanMessage(
                        `[Tool Result for ${toolCall.name}]: ${JSON.stringify(apiResult, null, 2)}`
                    )
                );

                // Check if this was send_notification
                if (toolCall.name === "send_notification") {
                    notificationSent = true;
                    console.log(
                        `  [${agent.agentName}] 🔔 Notification sent → will interrupt for user input`
                    );
                    // Don't iterate further — we'll interrupt after this node
                    break;
                }
            }

            return { messages: newMessages, notificationSent };
        });
    }

    // ┌──────────────────────────────────────────────────────┐
    // │  NODE: waitForInput                                   │
    // │  Calls interrupt() to PAUSE the graph.                │
    // │  When resumed via Command({ resume: ... }), the       │
    // │  user's input is added as a HumanMessage.             │
    // │  No side effects before interrupt → safe to re-run.   │
    // └──────────────────────────────────────────────────────┘
    graph.addNode("waitForInput", (state) => {
        console.log("  [waitForInput] ⏸ Interrupting — awaiting user input...");

        // interrupt() throws GraphInterrupt on first call (pausing the graph).
        // On resume, it returns the value passed via Command({ resume: ... }).
        const userResponse = interrupt({
            type: "awaiting_user_input",
            reason:
                "The notification agent requested additional information from the user.",
        });

        console.log(
            "  [waitForInput] ▶ Resumed with user input:",
            userResponse
        );

        const userMsg =
            typeof userResponse === "string"
                ? userResponse
                : JSON.stringify(userResponse);

        return {
            messages: [
                new HumanMessage(
                    `[User provided additional information]: ${userMsg}`
                ),
            ],
            notificationSent: false, // reset flag
        };
    });



    // ═══════════════════════════════════════════════════════
    //  EDGES
    // ═══════════════════════════════════════════════════════

    // START → supervisor
    graph.addEdge(START, "supervisor");

    // supervisor → agent, waitForInput, or END
    const routeMap = {};
    for (const agent of activeAgents) {
        routeMap[agent.agentName] = agent.agentName;
    }
    routeMap["waitForInput"] = "waitForInput";
    routeMap["FINISH"] = END;

    graph.addConditionalEdges(
        "supervisor",
        (state) => state.nextAgent || "FINISH",
        routeMap
    );

    // Each agent → waitForInput (if notification sent) or supervisor
    for (const agent of activeAgents) {
        graph.addConditionalEdges(
            agent.agentName,
            (state) => state.notificationSent ? "waitForInput" : "supervisor",
            {
                waitForInput: "waitForInput",
                supervisor: "supervisor",
            }
        );
    }

    // waitForInput → supervisor (after user provides info)
    graph.addEdge("waitForInput", "supervisor");

    // ═══════════════════════════════════════════════════════
    //  COMPILE with MemorySaver checkpointer
    // ═══════════════════════════════════════════════════════

    return graph.compile({ checkpointer });
}

// ═══════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════

function generateRunId() {
    return `run_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function getCompiledGraph(flowId) {
    if (!graphCache.has(flowId)) {
        graphCache.set(flowId, buildGroupChatGraph(flowId));
    }
    return graphCache.get(flowId);
}

function formatMessage(msg) {
    return {
        role: msg._getType?.() || msg.type || "unknown",
        content: msg.content || "",
        name: msg.name || undefined,
    };
}

function isInterruptError(err) {
    return (
        err.name === "GraphInterrupt" ||
        err.constructor?.name === "GraphInterrupt" ||
        (err.message && err.message.includes("GraphInterrupt"))
    );
}

function getInterruptData(snapshot) {
    return (snapshot?.tasks || [])
        .flatMap((t) => t.interrupts || [])
        .map((i) => i.value);
}

// ═══════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════

/**
 * Common streaming logic for start and resume
 */
async function streamGraph(graph, initialInput, config, onUpdate) {
    let finalState = null;
    let runId = config.configurable.thread_id;

    try {
        const stream = await graph.stream(initialInput, {
            ...config,
            streamMode: "values",
        });

        for await (const value of stream) {
            finalState = value;
            if (onUpdate) {
                // State 3: Trying to find information on my own (Keep loop running)
                onUpdate({
                    type: "update",
                    runId,
                    status: "running", 
                    messages: (value.messages || []).map(formatMessage),
                    nextAgent: value.nextAgent,
                });
            }
        }
    } catch (err) {
        if (isInterruptError(err)) {
            // State 2: Waiting for information from User (End technical loop, wait for resume)
            let interruptData = [];
            try {
                const snapshot = await graph.getState(config);
                interruptData = getInterruptData(snapshot);
                finalState = snapshot.values;
                await graph.updateState(config, { status: "interrupted" });
            } catch { }

            console.log(`\n⏸ [State: Waiting for User] RunID: ${runId}`);

            const lastMsg = finalState?.messages?.[finalState.messages.length - 1];
            const reason = lastMsg?.content || finalState?.detailedStatus || "Action Required";

            const result = {
                configId: runId, // mapped for persistence
                status: "interrupted",
                detailedStatus: finalState?.detailedStatus || "PendingWaitingForInput",
                reason: reason,
                interruptData,
                messages: (finalState?.messages || []).map(formatMessage),
            };
            if (onUpdate) onUpdate({ type: "result", ...result });
            return result;
        }

        // State 4: Error (End the loop)
        console.error(`\n❌ [State: Error] RunID: ${runId}\n`, err);
        try {
            await graph.updateState(config, { status: "error" });
        } catch {}
        
        const errorResult = {
            runId,
            status: "error",
            message: err.message,
            messages: (finalState?.messages || []).map(formatMessage),
        };
        if (onUpdate) onUpdate({ type: "error", ...errorResult });
        return errorResult;
    }

    // Check for manual interrupt return
    try {
        const snapshot = await graph.getState(config);
        const pendingInterrupts = getInterruptData(snapshot);
        if (pendingInterrupts.length > 0) {
            // State 2 (Manual check)
            const result = {
                runId,
                status: "interrupted",
                reason: "Awaiting user input — action required",
                interruptData: pendingInterrupts,
                messages: (snapshot.values?.messages || []).map(formatMessage),
            };
            await graph.updateState(config, { status: "interrupted" });
            if (onUpdate) onUpdate({ type: "result", ...result });
            return result;
        }
    } catch { }

    // State 1: Completion (End the loop)
    console.log(`\n✅ [State: Completion] RunID: ${runId}`);
    await graph.updateState(config, { status: "completed" });
    const finalResult = {
        configId: runId, // mapped for persistence
        status: "completed",
        detailedStatus: finalState?.detailedStatus || "complete",
        messages: (finalState?.messages || []).map(formatMessage),
    };
    if (onUpdate) onUpdate({ type: "result", ...finalResult });
    return finalResult;
}

/**
 * Start a new group chat run for a given flow.
 */
async function startGroupChat(flowId, prompt, onUpdate) {
    const runId = generateRunId();
    const graph = getCompiledGraph(flowId);
    const flow = flowsConfig.flows.find((f) => f.flowID === flowId);
    const gcNode = flow?.nodes.find((n) => n.type === "groupchat");

    const config = {
        configurable: { thread_id: runId },
        recursionLimit: 80,
    };

    console.log(`\n🚀 [Start] RunID: ${runId} | Flow: ${flowId}`);

    return streamGraph(
        graph,
        {
            messages: [new HumanMessage(prompt)],
            endCondition: gcNode?.endCondition || "",
            flowDirection: flow?.flowDescription || "",
            detailedStatus: "PendingLookingForSolution",
        },
        config,
        onUpdate
    );
}

/**
 * Resume a paused group chat run with user-provided input.
 */
async function resumeGroupChat(flowId, runId, userInput, onUpdate) {
    const graph = getCompiledGraph(flowId);
    const config = {
        configurable: { thread_id: runId },
        recursionLimit: 80,
    };

    // Prevent resuming completed runs
    try {
        const state = await graph.getState(config);
        if (state.values?.status === "completed") {
            const err = new Error("Cannot resume a completed flow.");
            if (onUpdate) onUpdate({ type: "error", message: err.message });
            throw err;
        }
    } catch (e) {
        if (e.message === "Cannot resume a completed flow.") throw e;
        // ignore other state errors (e.g. run doesn't exist yet)
    }

    console.log(`\n▶ [Resume] RunID: ${runId}`);

    return streamGraph(
        graph,
        new Command({ resume: userInput }),
        config,
        onUpdate
    );
}

// ═══════════════════════════════════════════════════════════
//  Exports
// ═══════════════════════════════════════════════════════════

module.exports = {
    startGroupChat,
    resumeGroupChat,
    buildGroupChatGraph,
    checkpointer,
};

const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json());

// Serve the flow initiator UI
app.use(express.static(path.join(__dirname, "..", "flowinitiator")));

/**
 * Endpoint to list available flows (still useful for UI init)
 */
app.get("/api/flows", (req, res) => {
    res.json(flowsConfig.flows.map(f => ({
        flowID: f.flowID,
        flowName: f.flowName,
        flowDescription: f.flowDescription
    })));
});

// WebSocket Configuration
wss.on("connection", (ws) => {
    console.log("🔌 New WebSocket connection established");

    ws.on("message", async (message) => {
        let data;
        try {
            data = JSON.parse(message);
        } catch (e) {
            return ws.send(JSON.stringify({ type: "error", message: "Invalid JSON" }));
        }

        const { type, flowId, prompt, runId, configId, userInput } = data;
        const targetId = runId || configId;

        const onUpdate = (update) => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ ...update, flowId }));
            }
        };

        if (type === "start") {
            try {
                await startGroupChat(flowId, prompt, onUpdate);
            } catch (err) {
                onUpdate({ type: "error", message: err.message });
            }
        } 
        
        else if (type === "resume") {
            try {
                await resumeGroupChat(flowId, targetId, userInput, onUpdate);
            } catch (err) {
                onUpdate({ type: "error", message: err.message });
            }
        }
    });

    ws.on("close", () => {
        console.log("🔌 WebSocket connection closed");
    });
});

const PORT = 3002;
server.listen(PORT, () => {
    console.log(`\n🚀 Group Chat Engine (Streaming) listening at http://localhost:${PORT}`);
    console.log(`💻 Flow Initiator UI: http://localhost:${PORT}/index.html`);
});

/**
 * Enhanced logging for interrupts to ensure the UI knows it happened.
 * The return structures of startGroupChat and resumeGroupChat already 
 * include 'interrupted' status, which we send back via WebSocket.
 */

module.exports = {
    startGroupChat,
    resumeGroupChat,
    buildGroupChatGraph,
    checkpointer,
};