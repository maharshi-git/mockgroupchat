const { ChatOllama } = require("@langchain/ollama");
const { StateGraph, Annotation, MemorySaver } = require("@langchain/langgraph");
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");

/**
 * 1. Define the Graph State
 */
const GraphState = Annotation.Root({
    messages: Annotation({
        reducer: (x, y) => x.concat(y),
        default: () => [],
    }),
    next: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => "Researcher",
    }),
    summary: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => "",
    }),
});

/**
 * 2. Setup the LLM (Ollama with phi3:mini)
 */
const llm = new ChatOllama({
    model: "phi3:mini",
    temperature: 0,
});

/**
 * 3. Define the Agent Nodes
 */

// Researcher Agent
async function researcherNode(state) {
    console.log("--- RESEARCHER ACTING ---");
    const response = await llm.invoke([
        new SystemMessage("You are a Researcher. Provide 3 interesting facts about the user's topic. respond with only the facts."),
        ...state.messages
    ]);
    return {
        messages: [new AIMessage({ content: response.content, name: "Researcher" })],
        next: "Writer"
    };
}

// Writer Agent
async function writerNode(state) {
    console.log("--- WRITER ACTING ---");
    const response = await llm.invoke([
        new SystemMessage("You are a Writer. Take the facts provided by the Researcher and turn them into a polite 2-sentence paragraph."),
        ...state.messages
    ]);
    return {
        messages: [new AIMessage({ content: response.content, name: "Writer" })],
        next: "Reviewer"
    };
}

// Reviewer Agent
async function reviewerNode(state) {
    console.log("--- REVIEWER ACTING ---");
    const response = await llm.invoke([
        new SystemMessage("You are a Reviewer. Check the writer's paragraph for clarity. If it is good, just say 'APPROVED' followed by the final text. If not, suggest one improvement."),
        ...state.messages
    ]);
    return {
        messages: [new AIMessage({ content: response.content, name: "Reviewer" })],
        next: "__end__" 
    };
}

/**
 * 4. Construct the Graph
 */
const workflow = new StateGraph(GraphState)
    .addNode("Researcher", researcherNode)
    .addNode("Writer", writerNode)
    .addNode("Reviewer", reviewerNode)
    .addEdge("__start__", "Researcher")
    .addEdge("Researcher", "Writer")
    .addEdge("Writer", "Reviewer")
    .addEdge("Reviewer", "__end__");

/**
 * 5. Add Memory Saver
 */
const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });

/**
 * 6. Execution Helper (Example)
 */
async function runWorkflow(topic, threadId = "thread-1") {
    const config = { configurable: { thread_id: threadId } };
    
    console.log(`\n\nStarting flow for topic: ${topic}\n`);
    
    const stream = await graph.stream(
        { messages: [new HumanMessage(topic)] },
        config
    );

    for await (const chunk of stream) {
        const [nodeName, values] = Object.entries(chunk)[0];
        console.log(`\n[Node: ${nodeName}]`);
        const lastMsg = values.messages[values.messages.length - 1];
        if (lastMsg) {
            console.log(`Content: ${lastMsg.content.substring(0, 500)}`);
        }
    }
}

// Run it if this file is executed directly
if (require.main === module) {
    runWorkflow("Quantum Computing").catch(console.error);
}

module.exports = { graph };
