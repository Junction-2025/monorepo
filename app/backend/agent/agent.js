import { Agent, run } from '@openai/agents';

export default async function agent(data) {

    const DroneDetectionReport = z.object({
        type: z.string().describe('The type of drone detection, should be one of the following: "drone-detected"'),
        model: z.string(),
        rpm: z.string(),
        rotors: z.number(),
        description: z.string().describe('A freeform description of the drone detection, keep it short'),
        timestamp: z.string(),
    });

    const systemPrompt = 'Your job is to create reports based on drone detection data that is sent to you';

    const agent = new Agent({
        name: 'Drone Detection Agent',
        instructions: systemPrompt,
        model: 'gpt-4.1',
        outputType: DroneDetectionReport,
    });

    const userPrompt = `Create a report using this data: ${JSON.stringify(data)}`;

    const result = await run(
        agent,
        userPrompt,
    );

    console.log(result.finalOutput);
    return result.finalOutput;
}
