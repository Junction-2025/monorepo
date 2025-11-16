'use client'
import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { AlertCircle, DroneIcon } from 'lucide-react';
import { Radar } from 'lucide-react';
import { Drone } from 'lucide-react';
import { ShieldAlert } from 'lucide-react';

export default function Dashboard() {


    const [droneData, setDroneData] = useState(null);


    useEffect(() => {

        const eventSource = new EventSource("http://localhost:8000/events");

        eventSource.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);
            console.log(data);
            if (data.type === "drone-detected") {
                setDroneData(data);
            }
        });

        eventSource.addEventListener('error', () => {
            console.error("error");
            eventSource.close();
        });

        return () => eventSource.close();
    }, []);


    return (
        <div className="flex flex-col items-center justify-center h-screen w-screen p-4">
            <h1 className="text-4xl font-semibold m-6 mb-10">Drone Destroyer 5000</h1>
            <div className="flex h-full w-full items-center justify-center">
                {droneData &&
                    <Card className="w-90">
                        <CardHeader>
                            <CardTitle className="flex flex-row gap-2 items-center text-xl">
                                <Drone className="w-6 h-6 shrink-0" />
                                Detection
                            </CardTitle>
                            <CardDescription>
                                {droneData.timestamp}
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="text-sm border px-4 py-4 rounded-xl mx-4">
                            <ul className="flex flex-col list-disc list-inside gap-2">
                                <li>RPM: {droneData.rpm}</li>
                                <li>Model: {droneData.model}</li>
                                <li>Rotors: {droneData.rotors}</li>
                            </ul>
                        </CardContent>
                        <CardFooter className="flex text-sm text-muted-foreground items-start gap-2">
                            <ShieldAlert className="w-4 h-4 mt-1 shrink-0" />
                            <p className="flex-1">{droneData.description}</p>
                        </CardFooter>
                    </Card>
                }
            </div>
        </div>
    );
}