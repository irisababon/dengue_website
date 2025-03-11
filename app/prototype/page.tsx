'use client'

import { useState } from 'react'
import {
  Dialog,
  DialogPanel,
  Disclosure,
  DisclosureButton,
  DisclosurePanel,
  Popover,
  PopoverButton,
  PopoverGroup,
  PopoverPanel,
} from '@headlessui/react'
import {
  ArrowPathIcon,
  Bars3Icon,
  ChartPieIcon,
  CursorArrowRaysIcon,
  FingerPrintIcon,
  SquaresPlusIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { ChevronDownIcon, PhoneIcon, PlayCircleIcon } from '@heroicons/react/20/solid'
import Link from 'next/link';
import Image from 'next/image'
import { inter, notoSans } from '../ui/fonts'

export default function Page() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="flex flex-col h-screen">
      <header className="bg-white sticky top-0 drop-shadow-lg z-40">
        <nav aria-label="Global" className="mx-auto flex max-w-7xl items-center justify-between p-3.5 lg:px-8">
          <div className="flex lg:flex-1">
            <Link href="#" className="-m-1.5 p-1.5">
              <img
                alt=""
                src="/mosquito.png"
                className="h-20 w-auto"
              />
            </Link>
          </div>
          <div className="flex lg:hidden">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              className="-m-2.5 inline-flex items-center justify-center rounded-md p-2.5 text-gray-700"
            >
              <Bars3Icon aria-hidden="true" className="size-8 mr-3.5" />
            </button>
          </div>
          <PopoverGroup className="hidden lg:flex lg:gap-x-12 mr-8">
            <Link href="../#" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
              Home
            </Link>           
            <Link href="./prototype" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
              Prototype
            </Link>
            <Link href="./contact_us" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
              Contact Us
            </Link>
          </PopoverGroup>
        </nav>
        <Dialog open={mobileMenuOpen} onClose={setMobileMenuOpen} className="lg:hidden">
          <div className="fixed inset-0 z-10" />
          <DialogPanel className="fixed inset-y-0 right-0 z-50 w-full overflow-y-auto bg-white px-6 py-6 sm:max-w-sm sm:ring-1 sm:ring-gray-900/10">
            <div className="flex items-center justify-between">
              <Link href="#" className="-m-1.5 p-1.5">
                <img
                  alt=""
                  src="/mosquito.png"
                  className="h-10 w-auto mt-3"
                />
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
                    <Link href="../#" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
                    Home
                    </Link>           
                    <Link href="./get_involved" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
                    Prototype
                    </Link>
                    <Link href="./contact_us" className="text-base/6 font-semibold text-gray-900 hover:text-purple-700">
                    Contact Us
                    </Link>
                </div>
              </div>
            </div>
          </DialogPanel>
        </Dialog>
      </header>
      <div className='overflow-scroll'>
          <div className='p-10'>
                <p>Prototype Page Placeholder</p>
          </div>
      </div>
    </div>
  )
}
