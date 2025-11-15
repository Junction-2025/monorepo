# Frontend

Next.js frontend application with React and Tailwind CSS.

## Prerequisites

- Node.js (v18 or higher recommended)
- npm or yarn

## Installation

1. Navigate to the frontend directory:
```bash
cd app/frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Development Mode

Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

Open [http://localhost:3000](http://localhost:3000) in your browser to see the result.

The page auto-updates as you edit the files.

### Production Build

Build the application for production:
```bash
npm run build
```

Start the production server:
```bash
npm start
```

### Linting

Run ESLint to check code quality:
```bash
npm run lint
```

## Project Structure

```
frontend/
├── src/
│   ├── app/              # Next.js app router pages
│   │   ├── page.js       # Home page
│   │   ├── dashboard/    # Dashboard page
│   │   └── layout.js     # Root layout
│   ├── components/       # React components
│   │   └── ui/           # UI components (shadcn/ui)
│   ├── http/             # HTTP client utilities
│   └── lib/              # Utility functions
├── public/               # Static assets
└── package.json
```

## Features

- Next.js 16 with App Router
- React 19
- Tailwind CSS 4
- Lucide React icons
- shadcn/ui components

## Environment Variables

Create a `.env.local` file for environment-specific configuration:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Learn More

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
