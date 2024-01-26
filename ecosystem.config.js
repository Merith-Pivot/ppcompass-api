module.exports = {
  apps: [
    {
      name: 'ppc-api', 
      script: 'uvicorn',
      args: 'app.main:app --reload', 
      autorestart: true,
      watch: true,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'development',
      },
      env_production: {
        NODE_ENV: 'production',
      },
    },
  ],
};
