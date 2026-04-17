const { ChatOllama } = require("@langchain/ollama");
const { StateGraph, Annotation, MemorySaver } = require("@langchain/langgraph");
const { HumanMessage, SystemMessage, AIMessage } = require("@langchain/core/messages");


const agents = require("./agents.json");
const flows = require("./flows.json");
// console.log(flows)




(
    async () => {
        const flowId = "mnyygg2y9zmjbq";
        let agentList = flows.flows.find((flow) => flow.flowID === flowId).nodes.find(x => x.type === "groupchat").agents;
        let flowCondition = flows.flows.find((flow) => flow.flowID === flowId).nodes.find(x => x.type === "groupchat").endCondition;
        // let agentDetails = agents.agents.find((agent) => agentList.includes(agent.agentName));
        let agentDetails = []
        agentList.forEach((agentName) => {
            agentDetails.push(agents.agents.find((agent) => agent.agentName === agentName));
        })


        // let agentDetails = agents.agents.filter((agent) => agentList.includes(agent.agentName));

        console.log(agentDetails)
    }
)()
