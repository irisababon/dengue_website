'use client'

import { useState, useEffect } from 'react'
import { Dialog, DialogPanel, Popover } from '@headlessui/react'
import { Bars3Icon, XMarkIcon } from '@heroicons/react/24/outline'
import Link from 'next/link'
import Image from 'next/image'
import Papa, { ParseResult } from 'papaparse'
import { Line, Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";

import { Poppins } from 'next/font/google'
const font2 = Poppins({ subsets: ['latin'], weight: ['400', '600', '700'] })

interface CSVRow {
  date: string;
  Cases: number;
  Rainfall: number;
  Temperature: number;
  RH: number;
  searches1: number;
  searches2: number;
}

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    Title,
    Tooltip,
    Legend
);

export default function Page() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [prediction, setPrediction] = useState<number[]>([])
  const [featuresNested, setFeaturesNested] = useState<number[][]>([])
  const [isGraphLoaded, setIsGraphLoaded] = useState(false)
  const [csvData, setCsvData] = useState<any[]>([])
  const [realTimeData, setRealTimeData] = useState<any>(null)

  useEffect(() => {
    document.title = "QC Dengue Prediction";
  }, []);

  useEffect(() => {
    fetch('/websiteSmooth.csv')
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse<CSVRow>(csvText, {
          header: true,
          dynamicTyping: true,
          complete: (results: ParseResult<CSVRow>) => {
            const data: CSVRow[] = results.data
            // Take the last 30 rows
            const last30 = data.slice(-30)
            // Map each row to an array of 6 features
            const nestedFeatures = last30.map((row: CSVRow) => [
              row.Cases,
              row.Rainfall,
              row.Temperature,
              row.RH,
              row.searches1,
              row.searches2
            ])
            setFeaturesNested(nestedFeatures)
          }
        })
      })
      .catch((error) => {
        console.error('Error fetching CSV:', error)
      })
  }, [])

  useEffect(() => {
    fetch('/websiteSmooth.csv')
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          complete: (result) => {
            const rows = result.data
            setCsvData(rows)
          },
        })
      })
  }, [])

  const fetchPrediction = async () => {
    if (featuresNested.length !== 30 || featuresNested.some(row => row.length !== 6)) {
      console.error('Features data is incomplete or malformed. Expected a 30x6 array, got', featuresNested)
      return;
    }
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: [featuresNested] })
      })
      const data = await response.json()
      console.log("Flask Response:", data);
      setPrediction(Array.isArray(data.prediction) ? data.prediction : [])
      
    } catch (error) {
      console.error('Error fetching prediction:', error)
    }
  }

  const updateData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/update', { method: 'GET' })
      const data = await response.json()
      console.log("Update Response:", data)
    } catch (error) {
      console.error('Error updating data:', error)
    }
  }

  useEffect(() => {
    if (featuresNested.length === 30) {
      fetchPrediction()
    }
  }, [featuresNested])

  const fetchRealTimeData = () => {
    if (csvData.length > 0) {
      const lastRow = csvData[csvData.length - 2]
      console.log("Last row of CSV data:", lastRow)
      return lastRow
    } else {
      console.error("CSV data is not yet loaded.")
      return null
    }
  }

  useEffect(() => {
    if (isGraphLoaded) {
      const lastRow = fetchRealTimeData()
      if (lastRow) {
        setRealTimeData(lastRow)
      }
    }
  }, [isGraphLoaded]) // Only run when graph is loaded

  const LineGraph = ({ prediction }: { prediction: number[] }) => {
    if (
      !Array.isArray(prediction[0]) ||
      !Array.isArray(prediction[1]) ||
      prediction[0].length !== prediction[1].length
    ) {
      return <p>Loading chart...</p>
    }
  
    const data = {
      labels: prediction[0], // Dates
      datasets: [
        {
          label: 'Predicted Cases',
          data: prediction[1], // Values
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.15,
        },
        {
          label: 'Actual Cases',
          data: [1,9,19,20,25,47,52,43,52,50,51,30,55,66,50,56,46,31,21,38,42,41,58,22,16,12,3],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.15,
        },
      ],
    }
  
    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Prediction for Dengue Cases',
          font: {
            size: 18,
            weight: 'normal', 
            family: 'Poppins, sans-serif',
          },
          padding: {
            top: 20,
            bottom: 0,
          },
        },
        legend: { 
            position: 'top' as const 
        },
      },
      scales: {
        y: {
          beginAtZero: true,
        }
      }
    }
    
    useEffect(() => {
      setIsGraphLoaded(true)
    }, [])
    return <Line options={options} data={data} />
  }

  interface BarGraphProps {
    realTimeData: string[]; 
  }

  const BarGraph = ({ realTimeData } : BarGraphProps) => {
    // Ensure realTimeData is passed and exists
    if (!realTimeData || realTimeData.length < 7) {
        return <div>Data is not available</div>;
    }

    // Use parseFloat and toFixed(0) for the data points
    const dataValues = [
        parseFloat(realTimeData[5]).toFixed(0), // Get and format the value for the first bar
        parseFloat(realTimeData[6]).toFixed(0), // Get and format the value for the second bar
        // You can add more values as necessary for other bars
    ];

    // Data for the bar graph
    const data = {
        labels: ['dengue', 'dengue symptoms'], // X-axis labels
        datasets: [
            {
                label: 'Relative Search Interest (0-100)', // You can change this label based on context
                data: dataValues, // The formatted data
                backgroundColor: 'rgb(75, 192, 192, 0.2)', // Color of bars
                borderColor: 'rgb(75, 192, 192)', // Color of the bar borders
                borderWidth: 3, // Border width
            },
        ],
    };

    // Options for the bar chart
    const options = {
        responsive: true, 
        aspectRatio: 1.2,
        plugins: {
            tooltip: {
                enabled: true,
                callbacks: {
                    label: function (tooltipItem) {
                        // Custom tooltip label
                        return `Label: ${tooltipItem.label}, Value: ${tooltipItem.raw} units`;
                    },
                },
            },
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Keywords', 
                },
                ticks: {
                  align: 'center', 
                  maxRotation: 0,  
                  minRotation: 0,  
                },  
            },
            y: {
                title: {
                    display: true,
                    text: '', 
                },        
                beginAtZero: true,
                min: 0,         
                max: 100,
            },
        },
    };

    return (
        <div>
            <Bar
                data={data}
                options={options}
                aria-label="Bar chart representing real-time search interest"
                role="img" 
            />
        </div>
    );
  };
  
  return (
    <div className="flex flex-col h-screen bg-neutral-100">
      <header className="bg-white sticky top-0 w-[94%] ml-[3%] mr-[3%] mt-[1.5rem] drop-shadow-lg rounded-[2.5rem] z-40">
        <nav aria-label="Global" className="mx-auto flex items-center justify-between p-3.5 lg:px-8">
          <div className="flex lg:flex-1">
            <Link href="#" className="-m-1.5 p-1.5">
              <img alt="Site logo" src="/placeholder_logo.png" className="h-20 w-auto" />
            </Link>
          </div>
          <div className="flex lg:hidden">
            <button
              type="button"
              aria-label="Open navigation menu"
              onClick={() => setMobileMenuOpen(true)}
              className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
            >
              <Bars3Icon aria-hidden="true" className="size-8 mr-3.5" />
            </button>
          </div>
          <Popover.Group className="hidden lg:flex lg:gap-x-14 mr-8">
            <Link href="../#" className="transition duration-450 ease-in-out text-base/6 font-semibold text-gray-900 hover:text-neutral-500">
              Home
            </Link>
            <Link href="./prototype" className="transition duration-450 ease-in-out text-base/6 font-semibold text-gray-900 hover:text-neutral-500">
              Prototype
            </Link>
            <Link href="./contact_us" className="transition duration-450 ease-in-out text-base/6 font-semibold text-gray-900 hover:text-neutral-500">
              Contact Us
            </Link>
          </Popover.Group>
        </nav>
        <Dialog open={mobileMenuOpen} onClose={setMobileMenuOpen} className="lg:hidden">
          <div className="fixed inset-0 z-10" />
          <Dialog.Panel className="fixed inset-y-0 right-0 z-50 w-full overflow-y-auto bg-white px-6 py-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10">
            <div className="flex items-center justify-between">
              <Link href="#" className="-m-1.5 p-1.5">
                <img alt="" src="/mosquito.png" className="h-10 w-auto mt-3" />
              </Link>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                className="-m-2.5 rounded-md p-2.5 text-gray-700"
              >
                <XMarkIcon aria-hidden="true" className="size-6" />
              </button>
            </div>
            <div className="mt-6 flow-root">
              <div className="-my-6 divide-y divide-gray-500/10">
                <div className="space-y-2 py-6">
                  <Link href="../#" className="text-base/6 font-semibold text-gray-900 hover:text-neutral-600">
                    Home
                  </Link>
                  <Link href="./get_involved" className="text-base/6 font-semibold text-gray-900 hover:text-neutral-600">
                    Prototype
                  </Link>
                  <Link href="./contact_us" className="text-base/6 font-semibold text-gray-900 hover:text-neutral-600">
                    Contact Us
                  </Link>
                </div>
              </div>
            </div>
          </Dialog.Panel>
        </Dialog>
      </header>
      <div className="ml-[5%] mr-[5%] mt-[3em]">
        <h1 className={`${font2.className} text-6xl mb-[10px]`}>Dashboard</h1>
        <p className="mb-[20px] italic opacity-70"> *Disclaimer: Project is currently in development; Actual case data is only available until January 27, 2025 </p>
        <div className = "flex flex-row">
          <div className="overflow-scroll flex-1 w-2/3 bg-white p-8 rounded-[1rem]">
              <div className="">
                  <button
                    onClick={updateData}
                    className="transition duration-250 ease-in-out bg-white border-[3px] border-neutral-400 hover:border-neutral-500 text-neutral-500 hover:text-neutral-600 font-bold py-2 px-4 rounded mb-4 mr-2"
                  >
                    Update Data
                  </button>
                <button
                  onClick={fetchPrediction}
                  className="transition duration-250 ease-in-out bg-white border-[3px] border-neutral-400 hover:border-neutral-500 text-neutral-500 hover:text-neutral-600 font-bold py-2 px-4 rounded mb-4 mr-2"
                >
                  Get Prediction
                </button>
                {/* {Array.isArray(prediction[0]) && Array.isArray(prediction[1]) &&
                prediction[0].length > 0 && prediction[0].length === prediction[1].length ? (
                prediction[0].map((date, idx) => (
                <p key={idx}> {date.toString()}: {prediction[1][idx]}</p>
                ))
                ) : (
                  <p>Loading predictions...</p>
                )} */}
              </div>
                <div className = "w-full w-3xl h-[450px]">
                  <LineGraph aria-live="Dengue forecast" prediction = {prediction}></LineGraph>
                </div>
          </div>

          <div className = "flex-2 p-4 ml-[20px] pl-[40px] w-1/3 bg-white p-8 rounded-[1rem]">
            <h2 className={`text-3xl mb-[20px] ${font2.className}`} >Real Time Data</h2>
            {realTimeData ? (
                <>
                  <div>Date: {realTimeData[0]}</div>
                  <div>Rainfall: {parseFloat(realTimeData[2]).toFixed(4)} mm</div>
                  <div>Temperature: {parseFloat(realTimeData[3]).toFixed(4)} ºC</div>
                  <div>Relative Humidity: {parseFloat(realTimeData[4]).toFixed(4)}%</div>
                  <div>Interest for 'dengue': {parseFloat(realTimeData[5]).toFixed(0)} out of 100</div>
                  <div>Interest for 'dengue symptoms': {parseFloat(realTimeData[6]).toFixed(0)} out of 100</div>

                  <div className = "w-full w-[300px] pt-[2rem]">
                    <BarGraph aria-live="Dengue forecast" realTimeData = {realTimeData}></BarGraph>
                  </div>
                </>
              ) : (
                <div className="text-neutral-500">Loading real-time data...</div>
              )}
          </div>
        </div>
      </div>
    </div>
  )
}
